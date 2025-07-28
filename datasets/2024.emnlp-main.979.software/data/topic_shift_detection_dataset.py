import torch
import os
import json
from random import randrange
from utils.util import load_json, get_tokenizer 
from copy import deepcopy
import torch.utils.data as data

class TopicShiftDetectionDataset(data.Dataset):
    def __init__(self, dataset_mode):
        super(TopicShiftDetectionDataset).__init__()
        self.dataset_mode = dataset_mode
        
    def initialize(self, opt):
        self.opt = opt
        data_path = self.get_paths(opt)
        self.data = self.preprocess_input(data_path)
        self.dataset_size = len(self.data)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = get_tokenizer(self.opt)
        self.tokenizer.add_special_tokens(
                {'additional_special_tokens' : ['<topic_shifted>']}
        )
        return len(self.tokenizer) 
        
    def get_paths(self, opt):
        data_path = os.path.join(opt.dataset_path, opt.dataset_name)
        return data_path
    
    def __getitem__(self,idx):
        item = {}
        example = self.data[idx]
        item['diag_idx'] = example.diag_idx
        item['input_seq'] = example.input_seq
        item['target_seq'] = example.target_seq
        
        return item
    
    def __len__(self):
        return self.dataset_size
    
    def preprocess_input(self, data_path):
        data = []
        total_idx = 0
        
        diags = load_json(data_path)
        for _sample in diags:
            diag = _sample['dialog']
            
            history = "Task: Determine if there is a shift in the topic aligned with <topic_shifted> in the last question."
            history += ' Question: ' + diag[0]['question'] + ' Answer: ' + diag[0]['answer']
            for diag_idx in range(1, len(diag)):
                history += ' Question: ' + diag[diag_idx]['question']
                if diag_idx in _sample['topic_shift']:
                    target_seq = "Topic shift occurred in the last question."
                else:
                    target_seq = "No topic shift in the last question."
                    
                instance = InstanceWrapper(total_idx, history, target_seq)
                total_idx +=1
                data.append(instance)
                history += ' Answer: ' + diag[diag_idx]['answer']       
                
        return data
    
    def collate_fn(self):
        collate_fn = Collator(self.tokenizer, self.opt.max_length)
        return collate_fn
    
class InstanceWrapper(object):
    def __init__(self, diag_idx, input_seq, target_seq):
        self.diag_idx = diag_idx
        self.input_seq = input_seq
        self.target_seq = target_seq

class Collator(object):
    def __init__(self, tokenizer, max_length = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __call__(self, batch):
        index = torch.tensor([b['diag_idx'] for b in batch])
        
        inputs = [b['input_seq'] for b in batch]
        target = [b['target_seq'] for b in batch]
        
        target = self.tokenizer(
            target,
            max_length = self.max_length if self.max_length > 0 else None,
            padding = 'max_length',
            return_tensors='pt',
            truncation=True if self.max_length > 0 else False
        )
        target_ids = target.input_ids
        target_ids[target_ids == self.tokenizer.pad_token_id] = -100
        
        inputs = self.tokenizer(
            inputs,
            max_length = self.max_length if self.max_length > 0 else None,
            padding= 'max_length',
            return_tensors = 'pt',
            truncation=True if self.max_length > 0 else False            
        )
        input_ids, input_mask = inputs.input_ids, inputs.attention_mask
        
        return index, input_ids, input_mask, target_ids