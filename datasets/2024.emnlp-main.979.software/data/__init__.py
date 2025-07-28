import torch.utils.data
from torch.utils.data import random_split
import importlib

def find_dataset_using_name(dataset_name):
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() :
            dataset = cls
            
    if dataset is None:
        raise ValueError("In %s.py, there should be a subclass of BaseDataset "
                         "with class name that matches %s in lowercase." %
                         (dataset_filename, target_dataset_name))

    return dataset

def create_dataloader(opt):
    dataset = find_dataset_using_name(opt.dataset)
    dataset = dataset(opt.dataset_mode) 
    opt.tokenizer_length = dataset.initialize(opt)
    print("dataset [%s] of size %d was created" %
          (type(dataset).__name__, len(dataset)))
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=True if opt.dataset_mode=='train' else False,
        num_workers=int(opt.nThreads),
        drop_last=opt.isTrain,
        collate_fn = dataset.collate_fn()
    )
    
    return dataloader