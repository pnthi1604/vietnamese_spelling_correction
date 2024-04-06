import torch
from torch.utils.data import Dataset
from underthesea import word_tokenize
from pyvi import ViTokenizer
import os
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import unidecode
import re
import random
from bs4 import BeautifulSoup
import contractions
import pandas as pd

class BilingualDataset(Dataset):

    def __init__(self, ds, src_lang, tgt_lang):
        super().__init__()
        self.ds = ds
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        print(f"{src_target_pair = }")
        print(self.src_lang)
        print(self.tgt_lang)
        src_text = src_target_pair[self.src_lang]
        tgt_text = src_target_pair[self.tgt_lang]

        return (src_text, tgt_text)

# create noise
# Lỗi tương đồng về mặt của chữ
chars_regrex = '[aàảãáạăằẳẵắặâầẩẫấậoòỏõóọôồổỗốộơờởỡớợeèẻẽéẹêềểễếệuùủũúụưừửữứựiìỉĩíịyỳỷỹýỵnvm]'
same_chars = {
    'a': ['á', 'à', 'ả', 'ã', 'ạ', 'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ'],
    'o': ['ó', 'ò', 'ỏ', 'õ', 'ọ', 'ô', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ', 'ơ','ớ', 'ờ', 'ở', 'ỡ', 'ợ', 'q'],
    'e': ['é', 'è', 'ẻ', 'ẽ', 'ẹ', 'ế', 'ề', 'ể', 'ễ', 'ệ', 'ê'],
    'u': ['ú', 'ù', 'ủ', 'ũ', 'ụ', 'ứ', 'ừ', 'ử', 'ữ', 'ự', 'ư'],
    'i': ['í', 'ì', 'ỉ', 'ĩ', 'ị'],
    'y': ['ý', 'ỳ', 'ỷ', 'ỹ', 'ỵ', 'v'],
    'n': ['m'],
    'v': ['y'],
    'm': ['n'],
}

def _char_regrex(text):
    match_chars = re.findall(chars_regrex, text)
    return match_chars

def _random_replace(text, match_chars):
    replace_char = match_chars[np.random.randint(low=0, high=len(match_chars), size=1)[0]]
    insert_chars = same_chars[unidecode.unidecode(replace_char)]
    insert_char = insert_chars[np.random.randint(low=0, high=len(insert_chars), size=1)[0]]
    text = text.replace(replace_char, insert_char, 1)

    return text

def replace_face_char(text):
    match_chars = _char_regrex(text)
    if len(match_chars) == 0:
        return text
    text = _random_replace(text, match_chars)
    return text
# End: lỗi tương đồng về mặt của chữ

# Lỗi tương đồng về âm thanh khi đọc
three_char = {
    "ngh": ["gh"]
}
two_char = {
    "ng": ["g", "n"],
    "gi": ["d", "v"],
    "gh": ["g"],
    "ch": ["tr", "c"],
    "tr": ["ch"],
    "ph": ["p"],
    "qu": ["q"],
    "kh": ["k"],
    "iê": ["i"],
    "iế": ["í"],
    "iệ": ["ị"],
    "iề": ["ì"],
    "iể": ["ỉ"],
    "iễ": ["ĩ"],
}
one_char = {
    "g": ["gh", "r"],
    "r": ["gh", "g"],
    "d": ["gi", "v"],
    "k": ["kh"],
    "ph": ["p"],
    "p": ["ph"],
    "q": ["qu"],
    "s": ["x"],
    "x": ["s"],
    "l": ["n"],
    "n": ["l"],
    "v": ["gi", "d"],
    "y": ["i"],
    "ý": ["í"],
    "ỳ": ["ì"],
    "ỷ": ["ỉ"],
    "ỹ": ["ĩ"],
    "ỵ": ["ị"],
    "i": ["y"],
    "í": ["ý"],
    "ì": ["ỳ"],
    "ỉ": ["ỷ"],
    "ĩ": ["ỹ"],
    "ị": ["ỵ"],
}

def replace_sound_char(word):
    n_chars = [three_char, two_char, one_char]
    for n_char in n_chars:
        for char in n_char:
            if char in word:
                return word.replace(char, random.choice(n_char[char]))
    return word
# End: lỗi tương đồng về mặt âm thanh khi đọc

# Lỗi gõ không dấu
def replace_no_accent_char(word):
    return unidecode.unidecode(word)
# End: lỗi gõ không dấu

# Lỗi gõ lặp lại chữ
def multi_word(word):
    return f"{word} {word}"
# End: lỗi gõ lặp lại chữ

# Lỗi mất chữ
def remove_word(word):
    return ""
# End: lỗi mất chữ

# create noise function
def random_ratio(array: list[int]=[], ratios: float=0.9, size: int=0):
    # print(f"{ratios = }")
    if len(array) != len(ratios):
        raise ValueError("The length of array and ratios must be the same")
    result_array = [-1] * len(array)
    for i in range(len(array)):
        num_element = int(size * ratios[i])
        sub_array = [array[i]] * num_element
        result_array += sub_array
    # print(f"{result_array = }")
    for i in range(len(result_array)):
        if result_array[i] == -1:
            result_array[i] = array[-1]
    np.random.shuffle(result_array)
    return result_array

def create_noise_random_choice(vi_sent):
    words = vi_sent.split()
    funcs_noise = [replace_face_char, replace_sound_char, replace_no_accent_char, remove_word, multi_word]
    if len(words) <= 1:
        return vi_sent
    func_noise = funcs_noise[random.randint(0, len(funcs_noise) - 1)]
    idx = random.randint(0, len(words) - 1)
    words[idx] = func_noise(words[idx])
    res = ""
    for i in range(len(words)):
        if words[i] == "":
            continue
        res = res + " " + words[i]
    return res.strip()

def create_noise(config, vi_sent):
    ratio = config["ratio"]
    face_char_ratio = round(config["face_char_ratio"] * ratio, 3)
    sound_char_ratio = round(config["sound_char_ratio"] * ratio, 3)
    no_accent_char_ratio = round(config["no_accent_char_ratio"] * ratio, 3)
    remove_word_ratio = round(config["remove_word_ratio"] * ratio, 3)
    multi_word_ratio = round(config["multi_word_ratio"] * ratio, 3)
    words = vi_sent.split()
    funcs_noise = [replace_face_char, replace_sound_char, replace_no_accent_char, remove_word, multi_word]
    ratios = [face_char_ratio,
              sound_char_ratio,
              no_accent_char_ratio,
              remove_word_ratio,
              multi_word_ratio,
              1 - (face_char_ratio + sound_char_ratio + no_accent_char_ratio + remove_word_ratio + multi_word_ratio)]
    rand_error = random_ratio(
        array=list(range(len(ratios))),
        ratios=ratios,
        size = len(words)
    )
    # print(f"{ rand_error = }")
    res = ""
    for i in range(len(words)):
        if rand_error[i] == len(ratios) - 1:
            res = res + " " + words[i]
        else:
            # print(f"{rand_error[i] = }")
            change_word = (funcs_noise[rand_error[i]])(words[i])
            if change_word != "":
                res = res + " " + change_word
    return res.strip()
# end create noise

def clean_data(text, lang):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = (text.lower()).replace(" '", "'")
    if lang == "en":
        text = contractions.fix(text)
    if lang == "vi":
        text = text.replace("you", "u")
    return text

def preprocess_function(config, example):
    output = {}
    vi_sent = clean_data(example["translation"][config["lang_tgt"]], lang=config["lang_tgt"])
    output[config["lang_tgt"]] = vi_sent
    output[config["lang_src"]] = vi_sent
    output[config["lang_src"]] = create_noise(
        config=config,
        vi_sent=output[config["lang_src"]]
    )
    if output[config["lang_tgt"]] == output[config["lang_src"]]:
        output[config["lang_src"]] = create_noise_random_choice(vi_sent=output[config["lang_src"]])
    return output

def load_data(config):
    data_path = f"{config['data_path']}"

    if "data" not in config:
        config["data"] = "mt_eng_vietnamese"
        config["subset"] = "iwslt2015-vi-en"
    elif "subset" not in config:
        config["subset"] = ""

    if not os.path.exists(data_path):
        dataset = load_dataset(config["data"], config["subset"])
        dataset.save_to_disk(data_path)
        print("\nĐã lưu dataset thành công!\n")

    dataset = load_from_disk(data_path)
    print("\nĐã load dataset thành công!\n")

    map_data_path = config["map_data_path"]    
    if not os.path.exists(map_data_path):
        dataset = dataset.map(
            lambda item: preprocess_function(config=config, example=item),
            remove_columns=dataset["train"].column_names,
        )

    return dataset

def filter_data(item, tokenizer_src, tokenizer_tgt, config):
  src_sent = item[config['lang_src']]
  tgt_sent = item[config['lang_tgt']]
  len_list_src_token = len(tokenizer_src.encode(src_sent).ids)
  len_list_tgt_token = len(tokenizer_tgt.encode(tgt_sent).ids)
  max_len_list = max(len_list_src_token, len_list_tgt_token)
  min_len_list = min(len_list_src_token, len_list_tgt_token)
  return max_len_list <= config["max_len"] - 4 and min_len_list > 4

def collate_fn(batch, tokenizer_src, tokenizer_tgt, pad_id_token):
    src_batch, tgt_batch, label_batch, src_text_batch, tgt_text_batch = [], [], [], [], []
    sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
    eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
    for src_text, tgt_text in batch:
        enc_input_tokens = tokenizer_src.encode(src_text).ids
        dec_input_tokens = tokenizer_tgt.encode(tgt_text).ids

        src = torch.cat(
            [
                sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                eos_token,
            ],
            dim=0,
        )

        # Add only <s> token
        tgt = torch.cat(
            [
                sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only </s> token
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                eos_token,
            ],
            dim=0,
        )

        src_batch.append(src)
        tgt_batch.append(tgt)
        label_batch.append(label)
        src_text_batch.append(src_text)
        tgt_text_batch.append(tgt_text)

    src_batch = pad_sequence(src_batch, padding_value=pad_id_token, batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=pad_id_token, batch_first=True)
    label_batch = pad_sequence(label_batch, padding_value=pad_id_token, batch_first=True)

    return {
        "encoder_input": src_batch,
        "decoder_input": tgt_batch,
        "label": label_batch,
        "src_text": src_text_batch,
        "tgt_text": tgt_text_batch,
    }

def get_dataloader(config, dataset, tokenizer_src, tokenizer_tgt):
    map_data_path = config["map_data_path"]                                                    
    if not os.path.exists(map_data_path):
        dataset = dataset.filter(lambda item: filter_data(item=item,
                                                          tokenizer_src=tokenizer_src,
                                                          tokenizer_tgt=tokenizer_tgt,
                                                          config=config))
        dataset_split = dataset["train"].train_test_split(train_size=config["train_size"], seed=42)
        dataset_split["validation"] = dataset_split.pop("test")
        dataset_split["test"] = dataset["test"]
        dataset_split["bleu_validation"] = dataset_split["validation"].select(range(config["num_bleu_validation"]))
        dataset_split["bleu_train"] = dataset_split["train"].select(range(config["num_bleu_validation"]))
        dataset_split.save_to_disk(map_data_path)
        # dataset.save_to_disk(map_data_path)
        print("\nĐã lưu map data thành công!\n")
    
    dataset = load_from_disk(map_data_path)
    print("\nĐã load map data thành công!\n")

    train_dataset = BilingualDataset(
        ds=dataset["train"],
        src_lang=config["lang_src"],
        tgt_lang=config["lang_tgt"],
    )

    validation_dataset = BilingualDataset(
        ds=dataset["validation"],
        src_lang=config["lang_src"],
        tgt_lang=config["lang_tgt"],
    )

    bleu_validation_dataset = BilingualDataset(
        ds=dataset["bleu_validation"],
        src_lang=config["lang_src"],
        tgt_lang=config["lang_tgt"],
    )

    bleu_train_dataset = BilingualDataset(
        ds=dataset["bleu_train"],
        src_lang=config["lang_src"],
        tgt_lang=config["lang_tgt"],
    )

    pad_id_token = tokenizer_tgt.token_to_id("[PAD]")
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config['batch_size_train'],
                                  shuffle=True, 
                                  collate_fn=lambda batch: collate_fn(batch=batch,
                                                                      pad_id_token=pad_id_token,
                                                                      tokenizer_src=tokenizer_src,
                                                                      tokenizer_tgt=tokenizer_tgt))
    validation_dataloader = DataLoader(validation_dataset, batch_size=config["batch_size_validation"],
                                       shuffle=False,
                                       collate_fn=lambda batch: collate_fn(batch=batch,
                                                                           pad_id_token=pad_id_token,
                                                                           tokenizer_src=tokenizer_src,
                                                                           tokenizer_tgt=tokenizer_tgt))
    bleu_validation_dataloader = DataLoader(bleu_validation_dataset, batch_size=1,
                                            shuffle=False,
                                            collate_fn=lambda batch: collate_fn(batch=batch,
                                                                                pad_id_token=pad_id_token,
                                                                                tokenizer_src=tokenizer_src,
                                                                                tokenizer_tgt=tokenizer_tgt))
    bleu_train_dataloader = DataLoader(bleu_train_dataset, batch_size=1,
                                            shuffle=False,
                                            collate_fn=lambda batch: collate_fn(batch=batch,
                                                                                pad_id_token=pad_id_token,
                                                                                tokenizer_src=tokenizer_src,
                                                                                tokenizer_tgt=tokenizer_tgt))

    return train_dataloader, validation_dataloader, bleu_validation_dataloader, bleu_train_dataloader

def check_test_item(src_sent, tgt_sent, tokenizer_src, tokenizer_tgt, config):
    eng_char = "wfjz"
    for char in eng_char:
        if char in src_sent or char in tgt_sent:
            return False
    len_list_src_token = len(tokenizer_src.encode(src_sent).ids)
    len_list_tgt_token = len(tokenizer_tgt.encode(tgt_sent).ids)
    max_len_list = max(len_list_src_token, len_list_tgt_token)
    min_len_list = min(len_list_src_token, len_list_tgt_token)
    return max_len_list <= config["max_len"] - 4 and min_len_list > 4

def get_dataloader_test(config, tokenizer_src, tokenizer_tgt):
    data_file_path = config["data_test"]

    dataset = pd.read_csv(data_file_path)
    data = []
    print(dataset)
    for i in range(len(dataset)):
        wrong_sent = dataset["wrong"][i]
        right_sent = dataset["right"][i]
        if not check_test_item(src_sent=wrong_sent, tgt_sent=right_sent, tokenizer_src=tokenizer_src, tokenizer_tgt=tokenizer_tgt, config=config):
            continue
        item = {
            config["lang_src"]: wrong_sent,
            config["lang_tgt"]: right_sent,
        }
        data.append(item)

    print(data)

    dataset = BilingualDataset(ds=data, src_lang=config["lang_src"], tgt_lang=["lang_tgt"])
    pad_id_token = tokenizer_tgt.token_to_id("[PAD]")
    test_dataloader = DataLoader(dataset, batch_size=1,
                                            shuffle=False,
                                            collate_fn=lambda batch: collate_fn(batch=batch,
                                                                                pad_id_token=pad_id_token,
                                                                                tokenizer_src=tokenizer_src,
                                                                                tokenizer_tgt=tokenizer_tgt))

    return test_dataloader