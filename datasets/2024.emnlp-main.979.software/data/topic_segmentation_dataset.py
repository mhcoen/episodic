import torch
import os
import json
from random import randrange
from utils.util import load_json, get_tokenizer 
from copy import deepcopy
import torch.utils.data as data

class TopicSegmentationDataset(data.Dataset):
    def __init__(self, dataset_mode):
        super(TopicSegmentationDataset).__init__()
        self.dataset_mode = dataset_mode
        
    def initialize(self, opt):
        self.opt = opt
        
        data_path = self.get_paths(opt)
        self.data = self.preprocess_input(data_path)
        self.dataset_size = len(self.data)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = get_tokenizer(self.opt)
        
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
            #target
            target_seq = _sample['topic_shift']
            target_seq = ','.join(map(str, target_seq))
            
            #input
            #input_seq = ["Task: Segment the topic for each Question-Answer pair corresponding to the <topic> token."]
            input_seq = ["In the provided dialog below, identify the sections where topic shifts occur. Output the indices where the topics change, separated by spaces."]
            for turn_idx in range(len(diag)):
                input_seq.append(str(turn_idx) + ' Question: ' + diag[turn_idx]['question'] + ' Answer: ' + diag[turn_idx]['answer'])
            input_seq = ' '.join(input_seq)
            
            instance = InstanceWrapper(total_idx, input_seq, target_seq)
            total_idx +=1 
            data.append(instance)
        
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
        diag_idx = torch.tensor([b['diag_idx'] for b in batch])
        
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
        
        return diag_idx, input_ids, input_mask, target_ids