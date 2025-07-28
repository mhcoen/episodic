import random
import json
import os, sys
import torch
import numpy as np
import data
import models
from config import TestOptions
import torch.nn as nn
from utils.util import random_seed, get_tokenizer, save_json
from transformers import logging
import warnings
from tqdm import tqdm
#from utils.metrics import calc_scores

warnings.filterwarnings(action='ignore') 

#logging
logging.set_verbosity_warning()
logging.set_verbosity_error()

#conifg
opt = TestOptions().parse()
random_seed(opt.seed_num)

#data loader
test_dataloader = data.create_dataloader(opt)

#model
model = models.create_model(opt)
model_name = (type(model).__name__) #T5, FLAN_T5, T0

#GPU_setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if len(opt.gpu_ids)>1:
    model = nn.DataParallel(model, device_ids = opt.gpu_ids)
model.to(device)

#Load model_state
PATH = os.path.join(opt.checkpoints_dir,opt.experiment_name,"%s.pt"%(opt.experiment_name))
model.load_state_dict(torch.load(PATH))

#tokenizer
tokenizer = get_tokenizer(opt)
    
ref, hypo = {}, {}
cnt = 0
for batch_idx, batch in tqdm(enumerate(test_dataloader)):
    model.eval()
    with torch.no_grad():
        if len(opt.gpu_ids)>1:
            output_sequences = model.module.generate(
                input_ids=batch[1].cuda(),
                attention_mask=batch[2].cuda(),
                do_sample=False,  # disable sampling to test if batching affects output
            )

        else:
            output_sequences = model.generate(
                input_ids=batch[1].cuda(),
                attention_mask=batch[2].cuda(),
                do_sample=False,  # disable sampling to test if batching affects output
            )
        
        target = batch[3]
        target[target == -100] = tokenizer.pad_token_id
        batch_ref = tokenizer.batch_decode(target, skip_special_tokens=True)
        batch_hyp = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

    for i in range(len(target)):
        ref[cnt] = [batch_ref[i]]
        hypo[cnt] = [batch_hyp[i]]
        cnt+=1            

#save_results
save_dir = './results/'
if opt.para:
    file_name = "%s_para.json"%(opt.experiment_name)
else:
    file_name = "%s.json"%(opt.experiment_name)
#f = open(os.path.join(save_dir, file_name), 'a')

save_json([ref,hypo], os.path.join(save_dir, file_name))
#f.write('\n\n')
#f.write(str(hypo))
#f.write('\n')
#scores = calc_scores(ref, hypo)
#print(scores)

#f.write(str(scores))
#f.close()
print('Testing was successfully finished.')
