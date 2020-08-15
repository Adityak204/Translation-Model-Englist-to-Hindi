import pandas as pd
import re
from tqdm import tqdm
import copy

# Loading hindi text
with open("C:\\Users\\Aditya Singh\\Desktop\\Deep Learning\\7. Language Modelling\\English to Hindi\\pruned_train.hi", "r", encoding='utf-8') as hindi_inp:
    _text = hindi_inp.read()
hindi_text = _text.split('\n')
# len(hindi_text)  -- 788099


# Loading english text
with open("C:\\Users\\Aditya Singh\\Desktop\\Deep Learning\\7. Language Modelling\\English to Hindi\\pruned_train.en", "r", encoding='utf-8') as eng_inp:
    _text = eng_inp.read()
eng_text = _text.split('\n')
# len(eng_text) -- 788099

# Removing Hindi sentences having english letter in it
ids_to_remove = {}
for _id, _t in tqdm(enumerate(hindi_text)):
    if len(re.findall(r'[a-zA-Z]', _t)) > 0:
        ids_to_remove[_id] = _t
        # ids_to_remove.append(_id)
    else:
        pass

# for _id, k in enumerate(ids_to_remove):
#     print(ids_to_remove[k])
#     if _id > 11:
#         break

ids_to_keep = [i for i in range(len(hindi_text)) if i not in ids_to_remove.keys()]

filtered_eng_text = []
filtered_hindi_text = []
for _id in tqdm(ids_to_keep):
    filtered_eng_text.append(eng_text[_id].lower())
    filtered_hindi_text.append(hindi_text[_id])


# Utils
def remove_sc(_line, lang="en"):
    # _line = copy.deepcopy(_line)
    if lang == "hi":
        _line = re.sub(r'[+\-*/@%>=;~{}×–`’"()_]', "", _line)
        _line = re.sub(r"(?:(\[)|(\])|(‘‘)|(’’))", '', _line)
    elif lang == "en":
        _line = re.sub(r'[+\-*/@%>=;~{}×–`’"()_|:]', "", _line)
        _line = re.sub(r"(?:(\[)|(\])|(‘‘)|(’’))", '', _line)
    return _line


# Downloading contractions
with open("Translation Model\\misc\\contractions.txt", "r") as inp_cont:
    contractions_list = inp_cont.read()
contractions_list = [re.sub('["]', '', x).split(":") for x in re.sub(r"\s+", " ", re.sub(r"(.*{)|(}.*)", '', contractions_list)).split(',')]
contractions_dict = dict((k.lower().strip(), re.sub('/.*', '', v).lower().strip()) for k, v in contractions_list)


# Cleaning Text
def clean_text(_text, lang="en"):
    # _text = copy.deepcopy(_text)
    if lang == "en":
        _text = remove_sc(_line=_text, lang=lang)
        for cn in contractions_dict:
            _text = re.sub(cn, contractions_dict[cn], _text)
    elif lang == "hi":
        _text = remove_sc(_line=_text, lang=lang)
    return _text


# Treating english sentences
clean_eng_text = []
for sent in tqdm(filtered_eng_text):
    clean_eng_text.append(clean_text(_text=copy.deepcopy(sent), lang="en"))


# Treating hindi sentences
clean_hindi_text = []
for sent in tqdm(filtered_hindi_text):
    clean_hindi_text.append(clean_text(_text=copy.deepcopy(sent), lang="hi"))


# Filtered Data
clean_data = pd.DataFrame({"eng_text": clean_eng_text, "hindi_text": clean_hindi_text})


# Train-Val split
from sklearn.model_selection import train_test_split
train_set, val_set = train_test_split(clean_data, test_size=0.1)
train_set.to_csv("train.csv", index=False)
val_set.to_csv("val.csv", index=False)


# Data Processing : tokenization, batching, padding, splits, data load
from torchtext.data import Field, TabularDataset, BucketIterator
import spacy
import sys
from indicnlp import common
from indicnlp.tokenize import indic_tokenize

# Settings for handling devnagri text
INDIC_NLP_LIB_HOME = r"C:/Users/Aditya Singh/Desktop/Deep Learning/7. Language Modelling/English to Hindi/Translation Model/indic_nlp_library"
INDIC_NLP_RESOURCES = r"C:/Users/Aditya Singh/Desktop/Deep Learning/7. Language Modelling/English to Hindi/Translation Model/indic_nlp_resources"
# Add library to Python path
sys.path.append(r'{}/src'.format(INDIC_NLP_LIB_HOME))
# Set environment variable for resources folder
common.set_resources_path(INDIC_NLP_RESOURCES)


# Settings for handling english text
spacy_eng = spacy.load("en_core_web_sm")


# Defining Tokenizer
def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


def tokenize_hindi(text):
    return [tok for tok in indic_tokenize.trivial_tokenize(text)]


# Defining Field
english_txt = Field(tokenize=tokenize_eng, lower=True)
hindi_txt = Field(tokenize=tokenize_hindi, init_token="<sos>", eos_token="<eos>")


# Defining Tabular Dataset
data_fields = [('eng_text', english_txt), ('hindi_text', hindi_txt)]
train_dt, val_dt = TabularDataset.splits(path='./', train='train.csv', validation='val.csv', format='csv', fields=data_fields)


# Building word vocab
english_txt.build_vocab(train_dt, max_size=10000, min_freq=2)
hindi_txt.build_vocab(train_dt, max_size=10000, min_freq=2)


# Defining Iterator
train_iter = BucketIterator(train_dt, batch_size=20, sort_key=lambda x: len(x.eng_text), shuffle=True)
val_iter = BucketIterator(val_dt, batch_size=20, sort_key=lambda x: len(x.eng_text), shuffle=True)



