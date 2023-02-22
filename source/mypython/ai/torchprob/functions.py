from functools import singledispatch
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from scipy import special as sp
from torch import Tensor

from . import config
from .distributions.base import Distribution, _eps
from .distributions.normal import Normal


@singledispatch
def KLdiv(p: Distribution, q: Distribution) -> Tensor:
    raise NotImplementedError()


@KLdiv.register
def _KLdiv(p: Normal, q: Normal) -> Tensor:
    if config.use_original:
        return KL_normal_normal(p.loc, p.scale, q.loc, q.scale)
    else:
        return torch.distributions.kl._kl_normal_normal(p, q)


def KL_normal_normal(
    mu_1: Tensor,
    sigma_1: Tensor,
    mu_2: Tensor,
    sigma_2: Tensor,
) -> Tensor:
    """
    References:
        https://github.com/emited/VariationalRecurrentNeuralNetwork/blob/0f23c87d11597ecf50ecbbf1dd37429861fd7aca/model.py#L172
    """

    kld_element = (
        ((mu_1 - mu_2).pow(2) + sigma_1.pow(2)) / sigma_2.pow(2)
        + 2 * torch.log(sigma_2 + _eps)
        - 2 * torch.log(sigma_1 + _eps)
        - 1
    )
    return 0.5 * kld_element


class log:
    """
    log p(x | cond)

    Match the look of the formula

    Examples:
        math : log (y | x)
        impl : tp.log(prob, y).given(x).mean()
    """

    def __init__(self, p: Distribution, *x: Tensor) -> None:
        self._p = p
        self._x = x

    def given(self, *cond_vars: Tensor, **cond_vars_k: Tensor):
        return self._p.given(*cond_vars, **cond_vars_k).log_prob(*self._x)
