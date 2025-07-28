import sys
import argparse
import os
from utils.util import *
import torch
import data
import models
import pickle

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # experiment specifics
        parser.add_argument('--experiment_name', type=str, default='MP2D_random%s', help='name of the experiment.')
        parser.add_argument('--gpu_ids', type=str, default='0,1,2,3', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--device', type=str, default='cuda', help='cpu if not')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--seed_num', type=int, default = 2024, help='random_seed_num')
        
        # model specifics
        parser.add_argument('--model', type=str, default='T5', help='which model to use T5 or Flan_T5 or T0') 
        parser.add_argument('--T5_model', type=str, default ='t5-base', help='t5_model, "google/t5-v1_1-xl"')
        parser.add_argument('--GPT2_model', type=str, default ='gpt2', help='gpt2_model')
        parser.add_argument('--Auto_model', type=str, default =  "google/flan-t5-base", help='auto_model, "google/flan-t5-xl", "google/t5-efficient-xxl", "https://huggingface.co/bigscience/T0_3B"')
        parser.add_argument('--tokenizer_length', type=int, default = 0, help='gpt2_tokenizer_length')
        parser.add_argument('--batchSize', type=int, default = 4, help='input batch size, 32') 
        parser.add_argument('--max_length', type=int, default = 512, help='max_sentence_length')

        # dataset specifics
        parser.add_argument('--dataset', type=str, default='topic_segmentation', help='dataset .py __name__, topic_segmentation, topic_shift_detection') 
        parser.add_argument('--dataset_path', type=str, default='./datasets/', help='dataset_path')
        parser.add_argument('--dataset_name', type=str, default='MP2D_random_%s.json', help='dataset_name') 
        parser.add_argument('--nThreads', default=8, type=int, help='# threads for loading data')
        parser.add_argument('--max_dataset_size', type=int, default=sys.maxsize, help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        #parser.add_argument('--load_from_opt_file', action='store_true', help='load the options from checkpoints and use that as default')
        
        self.initialized = True
        return parser

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def option_file_path(self, opt, makedir=False):
        expr_dir = os.path.join(opt.checkpoints_dir, opt.experiment_name)
        if makedir:
            mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt')
        return file_name

    def save_options(self, opt):
        file_name = self.option_file_path(opt, makedir=True)
        with open(file_name + '.txt', 'wt') as opt_file:
            for k, v in sorted(vars(opt).items()):
                comment = ''
                default = self.parser.get_default(k)
                if v != default:
                    comment = '\t[default: %s]' % str(default)
                opt_file.write('{:>25}: {:<30}{}\n'.format(str(k), str(v), comment))

        with open(file_name + '.pkl', 'wb') as opt_file:
            pickle.dump(opt, opt_file)

    def update_options_from_file(self, parser, opt):
        new_opt = self.load_options(opt)
        for k, v in sorted(vars(opt).items()):
            if hasattr(new_opt, k) and v != getattr(new_opt, k):
                new_val = getattr(new_opt, k)
                parser.set_defaults(**{k: new_val})
        return parser

    def load_options(self, opt):
        file_name = self.option_file_path(opt, makedir=False)
        new_opt = pickle.load(open(file_name + '.pkl', 'rb'))
        return new_opt

    def parse(self, save=False):

        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        
        # get the basic options
        opt, unknown = parser.parse_known_args()
        self.parser = parser
                    
        #experiment name
        ex_add_name = ''
        if opt.dataset == 'topic_segmentation':
            ex_add_name = '_seg'
        else:
            ex_add_name = '_dtc'
        opt.experiment_name = opt.experiment_name%(ex_add_name)
        opt.experiment_name += '_%s'%(opt.model)
        
        #dataset
        if opt.dataset_mode == 'train':
            opt.dataset_name = opt.dataset_name%(opt.dataset_mode)
        else:
            if opt.para:
                opt.dataset_name = opt.dataset_name%(opt.dataset_mode+"_para")
            else:
                opt.dataset_name = opt.dataset_name%(opt.dataset_mode)
        
        opt.isTrain = self.isTrain   # train or test
        self.print_options(opt)
        if opt.isTrain:
            self.save_options(opt)
            
        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        assert len(opt.gpu_ids) == 0 or opt.batchSize % len(opt.gpu_ids) == 0, \
            "Batch size %d is wrong. It must be a multiple of # GPUs %d." \
            % (opt.batchSize, len(opt.gpu_ids))

        self.opt = opt
        return self.opt
    

class TrainOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        # for displays
        parser.add_argument('--log_freq', type=int, default=100, help='frequency of saving log on tensorboardX')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console') 
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results') 
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--valid_freq', type=int, default=10000, help='frequency of performing validation') 
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.experiment_name]/web/')
        parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')
        parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')

        # for training
        parser.add_argument('--dataset_mode', type=str, default='train', help='train, test, etc')
        parser.add_argument('--isTrain', type=bool, default=True, help='Training mode')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--niter', type=int, default=10, help='# of iter at starting learning rate. This is NOT the total #epochs. Totla #epochs is niter + niter_decay')
        parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--fp16', action='store_true', help='use mixed precision package')
        parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of updates steps to accumulate before performing a backward/update pass')
        
        parser.add_argument('--optimizer', type=str, default='AdamW')
        parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
        parser.add_argument('--epsilon', type=float, default=1e-4, help='epsilon term of adam')
        parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
        parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm for gradient clipping')
        
        
        opt, _ = parser.parse_known_args()
        self.isTrain = True
        return parser    
    
class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--dataset_mode', type=str, default='test', help='train, test, etc')
        parser.add_argument('--para', action='store_true', help='if true, uses paraphrased test set')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes inputs in order to make batches, otherwise takes them randomly') #for test
        opt, _ = parser.parse_known_args()
        self.isTrain = False
        return parser    
