from pprint import pprint
from typing import List, Tuple, Union

import mypython.ai.torchprob as tp
import mypython.ai.torchprob.debug as tp_debug
import numpy as np
import torch
import torch.nn as nn
from mypython.terminal import Color
from torch import Tensor

from .core import NewtonianVAECell, NewtonianVAECellDerivation


class PurePControl:
    def __init__(
        self,
        Igoal: Tensor,
        alpha: float,
        cell: Union[NewtonianVAECell, NewtonianVAECellDerivation],
    ) -> None:
        self.alpha = alpha
        self.cell = cell
        self.x_goal = cell.q_encoder.cond(Igoal).rsample()

    def get_action(self, I_t):
        x_t = self.cell.q_encoder.cond(I_t).rsample()
        return self.get_action_from_x(x_t)

    def get_action_from_x(self, x_t):
        u_t = self.alpha * (self.x_goal - x_t)
        return u_t


class PControl(nn.Module):
    """
    TODO:

    Paper:
        P(u_t | x_t) = ...  (12)
    """

    def __init__(self) -> None:
        super().__init__()
