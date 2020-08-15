import re
import copy


# Importing contractions
with open("model_utility\\contractions.txt", "r") as inp_cont:
    contractions_list = inp_cont.read()
contractions_list = [re.sub('["]', '', x).split(":") for x in re.sub(r"\s+", " ", re.sub(r"(.*{)|(}.*)", '', contractions_list)).split(',')]
contractions_dict = dict((k.lower().strip(), re.sub('/.*', '', v).lower().strip()) for k, v in contractions_list)


def remove_sc(_line, lang="en"):
    # _line = copy.deepcopy(_line)
    if lang == "hi":
        _line = re.sub(r'[+\-*/#@%>=;~{}×–`’"()_]', "", _line)
        _line = re.sub(r"(?:(\[)|(\])|(‘‘)|(’’))", '', _line)
    elif lang == "en":
        _line = re.sub(r'[+\-*/#@%>=;~{}×–`’"()_|:]', "", _line)
        _line = re.sub(r"(?:(\[)|(\])|(‘‘)|(’’))", '', _line)
    return _line


def clean_text(_text, lang="en"):
    # _text = copy.deepcopy(_text)
    if lang == "en":
        _text = remove_sc(_line=_text, lang=lang)
        for cn in contractions_dict:
            _text = re.sub(cn, contractions_dict[cn], _text)
    elif lang == "hi":
        _text = remove_sc(_line=_text, lang=lang)
    return _text
