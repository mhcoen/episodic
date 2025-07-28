import torch
from utils.util import *
import torch.nn as nn

from transformers import AutoModelForSeq2SeqLM

class FlanT5(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FLAN_T5_model = AutoModelForSeq2SeqLM.from_pretrained(self.opt.Auto_model)#, device_map='auto')
        
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids != None:
            if input_ids.dim() == 3:
                input_ids = input_ids.view(input_ids.size(0), -1)
        if attention_mask != None:
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.view(attention_mask.size(0), -1)
            
        return self.FLAN_T5_model.forward( 
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

    def generate(self, input_ids, attention_mask, do_sample=False):
        if input_ids != None:
            if input_ids.dim() == 3:
                input_ids = input_ids.view(input_ids.size(0), -1)
        if attention_mask != None:
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.view(attention_mask.size(0), -1)

        return self.FLAN_T5_model.generate( 
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length= self.opt.max_length,
            do_sample = do_sample
        )
