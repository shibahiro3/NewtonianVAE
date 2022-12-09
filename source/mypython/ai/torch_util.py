import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim


def reproduce(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_module_params(module: nn.Module, grad=False):
    print("=== Model Parameters ===")
    print(type(module))
    for name, param in module.named_parameters(recurse=True):
        print(f"name: {name} | shape: {[*param.shape]}")
        if grad:
            if type(param.grad) == torch.Tensor:
                print(f"grad: \n{param.grad}\n")
            else:
                print(f"grad: {param.grad}")  # None


def _find_function(function_name: str):
    try:
        return getattr(torch, function_name)
    except:
        return getattr(F, function_name)
