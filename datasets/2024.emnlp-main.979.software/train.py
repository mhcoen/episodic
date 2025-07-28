import random
import json
import os, sys
import torch
import numpy as np
import data
import models
from config import TrainOptions
import torch.nn as nn
from utils.iter_counter import IterationCounter
from utils.visualizer import Visualizer
from utils.util import random_seed

from transformers import logging, get_scheduler
from torch.autograd import Variable
from torch.cuda.amp import GradScaler

from tqdm import tqdm

import warnings
warnings.filterwarnings(action='ignore') 

#logging
logging.set_verbosity_warning()
logging.set_verbosity_error()

#config
opt = TrainOptions().parse()
random_seed(opt.seed_num)

#data loader
train_dataloader = data.create_dataloader(opt)

#model
model = models.create_model(opt)
model_name = (type(model).__name__) #T5, FLAN_T5, T0

#continue_train
if opt.continue_train == True:
    PATH = os.path.join(opt.checkpoints_dir,"%s.pt"%(opt.experiment_name))
    if os.path.exists(PATH):
        model.load_state_dict(torch.load(PATH))
    else:
        print("Wrong PATH : check opt.checkpoints_dir and opt.experiment_name")

#GPU_setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if len(opt.gpu_ids)>1:
    model = nn.DataParallel(model, device_ids = opt.gpu_ids)
model.to(device)

#optimizer
if opt.optimizer == 'AdamW':
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.epsilon)
elif opt.optimizer == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.epsilon)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr)
    
#lr_scheduler
lr_scheduler = get_scheduler(name='linear', optimizer=optimizer, num_warmup_steps=0, num_training_steps=opt.niter * ((len(train_dataloader) // opt.batchSize) +1))

#mixed precision scaler
if opt.fp16:
    scaler = GradScaler()
    opt.scaler = scaler

iter_counter = IterationCounter(opt, len(train_dataloader))
visualizer = Visualizer(opt)
for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    total_train_loss = 0
    model.train()
    
    for batch_idx, batch in tqdm(enumerate(train_dataloader, start=iter_counter.epoch_iter)):
        batch = tuple(t.to(device) for t in batch)
        try:
            if opt.fp16:
                with torch.cuda.amp.autocast():
                    outputs = model(
                        input_ids=batch[1],
                        attention_mask=batch[2],
                        labels=batch[3],
                    )
                    loss = outputs.loss
            else:
                outputs = model(
                    input_ids=batch[1],
                    attention_mask=batch[2],
                    labels=batch[3],
                )
                loss = outputs.loss
            
            if len(opt.gpu_ids) > 1: 
                loss = loss.mean()
            if opt.gradient_accumulation_steps > 1:
                loss = loss / opt.gradient_accumulation_steps
                
            if opt.fp16:
                opt.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            total_train_loss += loss.item()
            
            if (batch_idx+1) % opt.gradient_accumulation_steps == 0:
                if opt.fp16:
                    opt.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)
                    opt.scaler.step(optimizer)
                    opt.scaler.update()
                    optimizer.zero_grad()
                    lr_scheduler.step()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()
            
            #visualizer
            if iter_counter.needs_save_log():
                visualizer.save_tf_log(total_train_loss, batch_idx, iter_counter.total_steps_so_far)
            
            if iter_counter.needs_printing():
                print("TOTAL STEP : %d ==> EPOCH #%d || STEP: %d "%(iter_counter.total_steps_so_far,epoch, batch_idx * opt.batchSize))
                visualizer.print_step(total_train_loss, batch_idx)
                
            if iter_counter.needs_saving():
                PATH = os.path.join(opt.checkpoints_dir,opt.experiment_name,"%s.pt"%(opt.experiment_name))
                torch.save(model.state_dict(), PATH)
            
            if iter_counter.needs_validation() and batch_idx > 0:
                print("Validation Start!")
                model.eval()
                #evaluate
                model.train()
            
            
        except Exception as e:
            if 'out of memory' in str(e):
                print('  >> WARN: Run out of memory, skipping batch!')
            else:
                print('  >> RuntimeError {}'.format(e))
        
        #iter_counter
        iter_counter.record_one_iteration()
                
    visualizer.print_epoch(total_train_loss, len(train_dataloader))
    
    #trainer.update_learning_rate(epoch)
    avg_train_loss = total_train_loss / len(train_dataloader)
    print("Training loss: {0:.5f}".format(avg_train_loss), end=' ')

    #logger
    iter_counter.record_epoch_end()    
        
    #epoch save model
    if (epoch+1) % (opt.save_epoch_freq+1) == 0 or epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        epoch_PATH = os.path.join(opt.checkpoints_dir,opt.experiment_name,"%s_epoch%02d.pt"%(opt.experiment_name, epoch))
        torch.save(model.state_dict(), epoch_PATH)

print('Training was successfully finished.')
