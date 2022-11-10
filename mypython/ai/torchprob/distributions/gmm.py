from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torch import Tensor

# from .normal import Normal
from torch.distributions.normal import Normal as torch_Normal
from typing_extensions import Self

from mypython.terminal import Color

from .base import Distribution, _to_optional_tensor


class GMM(Distribution):
    """
    Gaussian Mixture Model

    ∑k=1 K πk N(x | μk, Σk)

    "Normal (Gaussian) distribution" is not multivariate normal distribution.

    K: number of mixtures
    D: the number of features

    pi: shape (*, K)
    mus: shape (*, K, D)
    sigmas: shape (*, K, D)

    Ref:
        https://github.com/ctallec/world-models/blob/master/models/mdrnn.py
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, *cond_vars: Tensor, **cond_vars_k: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError()
        pi: Tensor
        mus: Tensor
        sigmas: Tensor
        return pi, mus, sigmas

    def cond(self, *cond_vars: Tensor, **cond_vars_k: Tensor) -> Self:
        self._log_pi, self._mus, self._sigmas = self(*cond_vars, **cond_vars_k)

        assert torch.isclose(self._log_pi.sum(dim=-1), 1).all()
        # BS = self._pi.shape[0]
        # K = self._pi.shape[-1]
        # assert self._mus.shape[:-1] == (BS, K)
        # assert self._mus.shape == self._sigmas.shape

    def log_p(self, x: Tensor) -> Tensor:
        assert self._log_pi is not None
        assert self._mus is not None
        assert self._sigmas is not None

        # return log_gmm


def log_gmm(x: Tensor, log_pi: Tensor, mus: Tensor, sigmas: Tensor):
    x = x.unsqueeze(-2)
    normal_dist = torch_Normal(mus, sigmas)
    g_log_probs = normal_dist.log_prob(x)
    g_log_probs = log_pi + torch.sum(g_log_probs, dim=-1)

    max_log_probs = torch.max(g_log_probs, dim=-1, keepdim=True)[0]
    g_log_probs = g_log_probs - max_log_probs

    g_probs = torch.exp(g_log_probs)
    probs = torch.sum(g_probs, dim=-1)

    log_prob = max_log_probs.squeeze() + torch.log(probs)

    return log_prob
