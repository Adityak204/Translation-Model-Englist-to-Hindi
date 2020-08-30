import torch
import math
import copy


"""
# Data Processing : tokenization, batching, padding, splits, data load
from torchtext.data import Field, TabularDataset, BucketIterator
import spacy
import sys
from indicnlp import common
from indicnlp.tokenize import indic_tokenize

# Settings for handling devnagri text
INDIC_NLP_LIB_HOME = r"/Translation Model/indic_nlp_library"
INDIC_NLP_RESOURCES = r"/Translation Model/indic_nlp_resources"
# Add library to Python path
sys.path.append(r'{}/src'.format(INDIC_NLP_LIB_HOME))
# Set environment variable for resources folder
common.set_resources_path(INDIC_NLP_RESOURCES)


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

sentence = "eating food timely is good for health."
src_field = english_txt
src_tokenizer = tokenize_eng
trg_field = hindi_txt
trg_vcb_sz = 10000
k = 5

"""


def beam_search(sentence, model, src_field, src_tokenizer, trg_field, trg_vcb_sz, k, max_ts=50):
    # Tokenize the input sentence
    sentence_tok = src_tokenizer(sentence)

    # Add <sos> and <eos> in beginning and end respectively
    sentence_tok.insert(0, src_field.init_token)
    sentence_tok.append(src_field.eos_token)

    # Converting text to indices
    src_tok = torch.tensor([src_field.vocab.stoi[token] for token in sentence_tok], dtype=torch.long).unsqueeze(0)
    trg_tok = torch.tensor([trg_field.vocab.stoi[trg_field.init_token]], dtype=torch.long).unsqueeze(0)

    # Setting 'eos' flag for target sentence
    eos = trg_field.vocab.stoi[trg_field.eos_token]

    # Store for top 'k' translations
    trans_store = {}

    store_seq_id = None
    store_seq_prob = None
    for ts in range(max_ts):
        if ts == 0:
            with torch.no_grad():
                out = model(src_tok, trg_tok)  # [1, trg_vcb_sz]
            topk = torch.topk(torch.log(torch.softmax(out, dim=-1)), dim=-1, k=k)
            seq_id = torch.empty(size=(k, ts + 2), dtype=torch.long)
            seq_id[:, :ts + 1] = trg_tok
            seq_id[:, ts + 1] = topk.indices
            seq_prob = topk.values
            if eos in seq_id[:, ts + 1]:
                trans_store[seq_prob[:, seq_id[:, ts + 1] == eos].squeeze()] = seq_id[seq_id[:, ts + 1] == eos, :].squeeze()
                store_seq_id = copy.deepcopy(seq_id[seq_id[:, ts + 1] != eos, :])
                store_seq_prob = copy.deepcopy(seq_prob[:, seq_id[:, ts + 1] != eos].squeeze())
            else:
                store_seq_id = copy.deepcopy(seq_id)
                store_seq_prob = copy.deepcopy(seq_prob)
        else:
            src_tok = src_tok.squeeze()
            src = src_tok.expand(size=(store_seq_id.shape[-2], len(src_tok)))
            with torch.no_grad():
                out = model(src, store_seq_id)
            out = torch.log(torch.softmax(out[:, -1, :], dim=-1))  # [k, trg_vcb_sz]
            all_comb = (store_seq_prob.view(-1, 1) + out).view(-1)
            all_comb_idx = torch.tensor([(x, y) for x in range(store_seq_id.shape[-2]) for y in range(trg_vcb_sz)])
            topk = torch.topk(all_comb, dim=-1, k=k)
            top_seq_id = all_comb_idx[topk.indices.squeeze()]
            top_seq_prob = topk.values
            seq_id = torch.empty(size=(k, ts + 2), dtype=torch.long)
            seq_id[:, :ts + 1] = torch.tensor([store_seq_id[i.tolist()].tolist() for i, y in top_seq_id])
            seq_id[:, ts + 1] = torch.tensor([y.tolist() for i, y in top_seq_id])
            seq_prob = top_seq_prob
            if eos in seq_id[:, ts + 1]:
                trans_store[seq_prob[seq_id[:, ts + 1] == eos].squeeze()] = seq_id[seq_id[:, ts + 1] == eos, :].squeeze()
                store_seq_id = copy.deepcopy(seq_id[seq_id[:, ts + 1] != eos, :])
                store_seq_prob = copy.deepcopy(seq_prob[seq_id[:, ts + 1] != eos].squeeze())
            else:
                store_seq_id = copy.deepcopy(seq_id)
                store_seq_prob = copy.deepcopy(seq_prob)
        if len(trans_store) == k:
            break

    if len(trans_store) == 0:
        best_translation = store_seq_id[0]
    else:
        best_translation = trans_store[max(trans_store)]
    return " ".join([trg_field.vocab.itos[w] for w in best_translation])
