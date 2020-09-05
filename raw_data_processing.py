import pandas as pd
import re
from tqdm import tqdm
import copy
from model_utility.data_prep_utils import remove_sc, clean_text
from sklearn.model_selection import train_test_split


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


ids_to_keep = [i for i in range(len(hindi_text)) if i not in ids_to_remove.keys()]
filtered_eng_text = []
filtered_hindi_text = []
for _id in tqdm(ids_to_keep):
    filtered_eng_text.append(eng_text[_id].lower())
    filtered_hindi_text.append(hindi_text[_id])


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


# Filtering data based on sentence length
clean_data["eng_len"] = clean_data.eng_text.str.count(" ")
clean_data["hindi_len"] = clean_data.hindi_text.str.count(" ")
small_len_data = clean_data.query('eng_len < 50 & hindi_len < 50')


# Train-Val split
# Full set
train_set, val_set = train_test_split(small_len_data.loc[:, ["eng_text", "hindi_text"]], test_size=0.1)
train_set.to_csv("train.csv", index=False)
val_set.to_csv("val.csv", index=False)

# Small set
small_data = small_len_data.loc[:, ["eng_text", "hindi_text"]].sample(n=150000)
train_set_sm, val_set_sm = train_test_split(small_data, test_size=0.3)
train_set_sm.to_csv("train_sm.csv", index=False)
val_set_sm.to_csv("val_sm.csv", index=False)

