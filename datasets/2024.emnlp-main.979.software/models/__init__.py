import torch
import torch.nn as nn
import importlib

def find_model_using_name(model_name):
    # Given the option --model [modelname], the file "models/modelname_model.py"  will be imported.
    model_filename = "models." + model_name
    modellib = importlib.import_module(model_filename)

    # In the file, the class called ModelNameModel() will be instantiated. It has to be a subclass of torch.nn.Module, and it is case-insensitive.
    model = None
    target_model_name = model_name.replace('_', '')
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, torch.nn.Module):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of torch.nn.Module with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model

def create_model(opt):
    model = find_model_using_name(opt.model)
    instance = model(opt)
    print("model [%s] was created" % (type(instance).__name__))
    
    # Get all of the model's parameters as a list of tuples.
    params = list(model(opt).named_parameters())
    print('The model has {:} different named parameters.\n'.format(len(params)))

    return instance

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)