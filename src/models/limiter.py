from numbers import Real
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import NumberType, Tensor, nn

# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/522

# getattr(limiter, name, lambda x: x)


def range01_sigmoid(x):
    """Return range: [0, 1]"""
    return torch.sigmoid(x)


def range01_tanh(x):
    """Return range: [0, 1]"""
    return 0.5 * torch.tanh(x) + 0.5


def range11_sigmoid(x):
    """Return range: [-1, 1]"""
    return 2 * torch.sigmoid(x) - 1


def range11_tanh(x):
    """Return range: [-1, 1]"""
    return torch.tanh(x)


def range05_sigmoid(x):
    """Return range: [-0.5, 0.5]"""
    return torch.sigmoid(x) - 0.5


def range05_tanh(x):
    """Return range: [-0.5, 0.5]"""
    return 0.5 * torch.tanh(x)
