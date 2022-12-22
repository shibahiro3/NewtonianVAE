from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
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

    def log_prob(self, x: Tensor) -> Tensor:
        """log-likelihood of Distribution
        log Dist(x | ・)
        """
        raise NotImplementedError()

    def sample(self) -> Tensor:
        raise NotImplementedError()

    def rsample(self) -> Tensor:
        raise NotImplementedError()

    def decode(self) -> Tensor:
        raise NotImplementedError()

    def dist_parameters(self) -> Tuple[Tensor]:
        raise NotImplementedError()

    @property
    def loc(self) -> Tensor:
        raise NotImplementedError()

    @property
    def scale(self) -> Tensor:
        raise NotImplementedError()


def to_optional_tensor(x: Union[None, int, float, Tensor]) -> Union[None, Tensor]:
    _type = type(x)
    if x is None:
        pass
    elif _type == int or _type == float:
        x = torch.tensor(x)
    else:
        assert _type == Tensor
    return x


# _eps = torch.finfo(torch.float).eps  # DO NOT USE
_eps = 0
