import os
import json
import numpy as np
import random
import torch
from transformers import T5Tokenizer, GPT2Tokenizer, AutoTokenizer

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
def read_json_lines(file_path):
    with open(file_path, "r") as f:
        lines = []
        for l in f.readlines():
            loaded_l = json.loads(l.strip("\n"))
            lines.append(loaded_l)
    return lines

def save_json(data, file_path):
    with open(file_path, "w", encoding='utf-8') as f:
        json.dump(data, f, cls=NpEncoder, ensure_ascii=False)
        
def save_json_pretty(data, file_path):
    """save formatted json, use this one for some json config files"""
    with open(file_path, "w") as f:
        f.write(json.dumps(data, indent=4, sort_keys=True))

def load_json(file_path):
    with open(file_path, "r", encoding='utf-8-sig') as f:
        return json.load(f)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    
def random_seed(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
def get_tokenizer(opt):
    if opt.model.startswith('T5'):
        tokenizer = T5Tokenizer.from_pretrained(opt.T5_model)
    elif opt.model.startswith('GPT2'):
        tokenizer = GPT2Tokenizer.from_pretrained(opt.GPT2_model)
        tokenizer.bos_token = '<|startoftext|>'
        tokenizer.eos_token = '<|endoftext|>'
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        tokenizer.add_special_tokens(
                {'additional_special_tokens' : ['<|user|>', '<|system|>','<|context|>','<|endofcontext|>']}
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(opt.Auto_model)
        
    return tokenizer
