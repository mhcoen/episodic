import json
import torch
import os, sys
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%y%m%d-%H%M")
        self.writer = SummaryWriter(log_dir='./runs/%s-%s'%(opt.experiment_name,formatted_datetime))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def save_tf_log(self, total_train_loss, current_step, total_step):
        self.writer.add_scalar('train loss', total_train_loss / (current_step+1), total_step)       
        
    def print_epoch(self, total_train_loss, data_size):
        print('Training Loss : {0:.3f}'.format(total_train_loss / data_size))

    def print_step(self, total_train_loss, step):
        print('Training Loss : {0:.3f}'.format(total_train_loss / (step+1)))
        
    def save_model(self, model, model_name, valid_dataloader, step):
        """
        model.eval()
        if...best < current torch.save ...
        model.train()
        """        
        pass
