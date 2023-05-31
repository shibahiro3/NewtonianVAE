import math
from numbers import Number, Real
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from typing_extensions import Self

from mypython.terminal import Color


class Distribution(nn.Module):

    ParamsReturnType = Union[Tensor, List[Tensor]]

    def __init__(self) -> None:
        super().__init__()
        self._cnt_given = 0  # Number of times given() has been called

        # self._Theta = ...  # Parameters of the distribution

    def forward(self, *cond_vars, **cond_vars_k) -> ParamsReturnType:
        """Compute Θ of p(x | Θ) from cond_vars and cond_vars_k"""
        raise NotImplementedError()

    def __call__(self, *args, **kwargs) -> ParamsReturnType:
        return super().__call__(*args, **kwargs)

    def given(self, *cond_vars, **cond_vars_k) -> Self:
        """__call__(forward) is called and the parameters of the distribution are saved"""
        raise NotImplementedError()
        return self

    def log_prob(self, x: Tensor) -> Tensor:
        """log-likelihood of Distribution
        log p(x | Θ)
        """
        raise NotImplementedError()

    def sample(self) -> Tensor:
        """Sampling with detached current graph"""
        raise NotImplementedError()

    def rsample(self) -> Tensor:
        """Sampling with attached current graph"""
        raise NotImplementedError()

    def decode(self) -> Tensor:
        """
        It is difficult to read from the program that the mathematical variable Θ represents an image, etc.
        Therefore, this function is defined as an alias for Θ.
        """
        raise NotImplementedError()

    def dist_parameters(self) -> ParamsReturnType:
        """Return Θ"""
        raise NotImplementedError()

    def clear_dist_parameters(self) -> None:
        raise NotImplementedError()

    @property
    def loc(self) -> Tensor:
        raise NotImplementedError()

    @property
    def scale(self) -> Tensor:
        raise NotImplementedError()

    # @property
    # def param_*(self) -> Tensor:
    #   Parameters specific to each distribution (Public)


def to_optional_tensor(x: Union[None, int, float, Tensor]) -> Optional[Tensor]:
    _type = type(x)
    if x is None:
        return None
    elif _type == int or _type == float:
        return torch.tensor(x)
    else:
        assert _type == Tensor
        return x


class ProbParamsValueError(Exception):
    pass
