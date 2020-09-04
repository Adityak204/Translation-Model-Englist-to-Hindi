# Basic packages
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

# Packages for data generator & preparation
from torchtext.data import Field, TabularDataset, BucketIterator
import spacy
import sys
from indicnlp import common
from indicnlp.tokenize import indic_tokenize

# Settings for handling devnagri text
# INDIC_NLP_LIB_HOME = r"/indic_nlp_library"
# INDIC_NLP_RESOURCES = r"/indic_nlp_resources"
INDIC_NLP_LIB_HOME = r"/Translation Model/indic_nlp_library"
INDIC_NLP_RESOURCES = r"/Translation Model/indic_nlp_resources"
# Add library to Python path
sys.path.append(r'{}/src'.format(INDIC_NLP_LIB_HOME))
# Set environment variable for resources folder
common.set_resources_path(INDIC_NLP_RESOURCES)


# Packages for model building & inferences
from model_pack.transformer import Transformer
from model_utility.translator import beam_search
from model_utility.utils import save_checkpoint

# Data Prep
# Settings for handling english text
spacy_eng = spacy.load("en_core_web_sm")


# Defining Tokenizer
def tokenize_eng(text):
    return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]


def tokenize_hindi(text):
    return [tok for tok in indic_tokenize.trivial_tokenize(text)]


# Defining Field
english_txt = Field(tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>")
hindi_txt = Field(tokenize=tokenize_hindi, init_token="<sos>", eos_token="<eos>")

# Defining Tabular Dataset
data_fields = [('eng_text', english_txt), ('hindi_text', hindi_txt)]
train_dt, val_dt = TabularDataset.splits(path='./', train='train.csv', validation='val.csv', format='csv', fields=data_fields)

# Building word vocab
english_txt.build_vocab(train_dt, max_size=10000, min_freq=2)
hindi_txt.build_vocab(train_dt, max_size=10000, min_freq=2)

# Training & Evaluation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_model = True

# Training hyperparameters
num_epochs = 1
learning_rate = 3e-4
batch_size = 32

# Defining Iterator
train_iter = BucketIterator(train_dt, batch_size=batch_size, sort_key=lambda x: len(x.eng_text), shuffle=True)
val_iter = BucketIterator(val_dt, batch_size=batch_size, sort_key=lambda x: len(x.eng_text), shuffle=True)

# Model hyper-parameters
src_vocab_size = len(english_txt.vocab)
trg_vocab_size = len(hindi_txt.vocab)
embedding_size = 512
num_heads = 8
num_layers = 3
dropout = 0.10
max_len = 1000
forward_expansion = 4
src_pad_idx = english_txt.vocab.stoi["<pad>"]
trg_pad_idx = 0

# Defining model & optimizer attributes
model = Transformer(src_vocab_size=src_vocab_size,
                    trg_vocab_size=trg_vocab_size,
                    src_pad_idx=src_pad_idx,
                    trg_pad_idx=trg_pad_idx,
                    embed_size=embedding_size,
                    num_layers=num_layers,
                    forward_expansion=forward_expansion,
                    heads=num_heads,
                    dropout=dropout,
                    device=device,
                    max_len=max_len).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, verbose=True)

pad_idx = hindi_txt.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
loss_tracker = []

for epoch in range(num_epochs):
    model.train()
    losses = []
    loop = tqdm(enumerate(train_iter), total=len(train_iter))
    for batch_idx, batch in loop:
        # Get input and targets and move to GPU if available
        # Switching axis because bucket-iterator gives output of size(seq_len,bs)
        inp_data = batch.eng_text.permute(-1, -2).to(device)
        target = batch.hindi_text.permute(-1, -2).to(device)

        # Forward prop
        output = model(inp_data, target[:, :-1])

        optimizer.zero_grad()
        loss = criterion(output.reshape(-1, trg_vocab_size), target[:, 1:].reshape(-1))
        losses.append(loss.item())

        # Back prop
        loss.backward()
        # Clipping exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        # Gradient descent step
        optimizer.step()

        # Update progress bar
        loop.set_postfix(loss=loss.item())

    train_mean_loss = sum(losses) / len(losses)
    scheduler.step(train_mean_loss)

    model.eval()
    val_losses = []
    with torch.no_grad:
        for val_batch_idx, val_batch in enumerate(val_iter):
            val_inp_data = val_batch.eng_text.permute(-1, -2).unsqueeze(0).to(device)
            val_target = val_batch.hindi_text.permute(-1, -2).unsqueeze(0).to(device)
            val_output = model(val_inp_data, val_target[:-1, :])
            val_loss = criterion(val_output.squeeze(), val_output.view(-1))
            val_losses.append(val_loss.item())
        val_mean_loss = sum(val_losses)/len(val_losses)

    loss_tracker.append(val_mean_loss)

    if epoch % 10 == 0:
        if save_model and val_mean_loss == np.min(loss_tracker):
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint)

    print(f"Epoch [{epoch}/{num_epochs}]: train_loss= {train_mean_loss}; val_loss= {val_mean_loss}")

