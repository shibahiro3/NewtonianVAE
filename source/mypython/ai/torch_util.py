import random
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn, optim

_NT = Union[np.ndarray, torch.Tensor]


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


def find_function(function_name: str):
    try:
        return getattr(torch, function_name)
    except:
        return getattr(F, function_name)


def swap01(x: _NT):
    assert x.ndim >= 3
    axis = (1, 0) + tuple(range(2, x.ndim))
    if type(x) == np.ndarray:
        return x.transpose(axis)
    elif type(x) == Tensor:
        return x.permute(axis)
    else:
        assert False
