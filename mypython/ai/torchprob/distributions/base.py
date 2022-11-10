from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torch import Tensor
from typing_extensions import Self

from mypython.terminal import Color


class Distribution(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._cnt = 0

    def forward(self, *cond_vars: Tensor, **cond_vars_k: Tensor):
        """Compute Θ of Dist(x | Θ) from Dist(x | cond_vars)"""
        raise NotImplementedError()
        return self

    def cond(self, *cond_vars: Tensor, **cond_vars_k: Tensor) -> Self:
        """Compute Θ of Dist(x | Θ) from Dist(x | cond_vars)"""
        raise NotImplementedError()
        return self

    def log_p(self, x: Tensor) -> Tensor:
        """log-likelihood of Distribution
        log Dist(x | ・)
        """
        raise NotImplementedError()

    def rsample(self) -> Tensor:
        raise NotImplementedError()

    def decode(self) -> Tensor:
        raise NotImplementedError()

    def get_dist_params(self) -> Tuple[Tensor]:
        raise NotImplementedError()

    @property
    def loc(self) -> Tensor:
        raise NotImplementedError()

    @property
    def scale(self) -> Tensor:
        raise NotImplementedError()


def _to_optional_tensor(x: Union[None, int, float, Tensor]) -> Union[None, Tensor]:
    _type = type(x)
    if x is None:
        pass
    elif _type == int or _type == float:
        x = torch.tensor(x)
    else:
        assert _type == Tensor
    return x
