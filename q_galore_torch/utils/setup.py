import numpy as np
import torch 
import os 
from .quantization import QGaLoreLinear

def saving_model_weight(model, path):
    """
    Save model weight to file
    """
    checkpoint = model.state_dict()
    for name, module in model.named_modules():
        if isinstance(module, QGaLoreLinear):
            checkpoint[name + '.weight'] = module.weight
            if module.bias is not None:
                checkpoint[name + '.bias'] = module.bias
            checkpoint[name + '.scales'] = module.weight.scales
            checkpoint[name + '.zeros'] = module.weight.zeros
            checkpoint[name + '.group_size'] = module.weight.group_size
            checkpoint[name + '.saved_data_dtype'] = module.weight.saved_data_dtype
            checkpoint[name + '.stochastic_round'] = module.weight.stochastic_round
    torch.save(checkpoint, path)

def load_model_weight(model, path):
    """
    Load model weight from file
    """
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)
    for name, module in model.named_modules():
        if isinstance(module, QGaLoreLinear):
            module.weight = checkpoint[name + '.weight']
            if module.bias is not None:
                module.bias = checkpoint[name + '.bias']
            module.weight.scales = checkpoint[name + '.scales']
            module.weight.zeros = checkpoint[name + '.zeros']
            module.weight.group_size = checkpoint[name + '.group_size']
            module.weight.saved_data_dtype = checkpoint[name + '.saved_data_dtype']
            module.weight.stochastic_round = checkpoint[name + '.stochastic_round']

    return model
