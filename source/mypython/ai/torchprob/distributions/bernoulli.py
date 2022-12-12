from typing import List, Optional, Tuple, Union

import torch
from torch import NumberType, Tensor, nn
from typing_extensions import Self

from mypython.terminal import Color

from .base import Distribution, _eps, _to_optional_tensor


class Bernoulli(Distribution):
    # torch.distributions.Bernoulli

    def __init__(self, mu: Union[None, NumberType, Tensor] = None) -> None:
        super().__init__()

        self._mu = _to_optional_tensor(mu)

    def forward(self, *cond_vars: Tensor, **cond_vars_k: Tensor) -> Tensor:
        """Compute μ of Bern(x | μ) from Bern(x | cond_vars)"""
        raise NotImplementedError()
        mu: Tensor
        return mu

    def cond(self, *cond_vars: Tensor, **cond_vars_k: Tensor) -> Self:
        self._mu = self(*cond_vars, **cond_vars_k)

        _check_mu = self._mu.nan_to_num()
        assert (
            (0 <= _check_mu).all() and (_check_mu <= 1).all()
        ).item(), "μ should be 0 <= μ <= 1 (Bernoulli distribution)"

        self._cnt += 1 if self._cnt < 1024 else 0
        return self

    def log_p(self, x: Tensor) -> Tensor:
        assert self._mu is not None
        assert x.shape == self._mu.shape
        return log_bernoulli(x, self._mu)

    def decode(self) -> Tensor:
        assert self._mu is not None
        return self._mu.detach()

    def get_dist_params(self) -> Tensor:
        assert self._mu is not None
        return self._mu

    @property
    def loc(self) -> Tensor:
        return self._mu

    @property
    def scale(self) -> Tensor:
        return torch.sqrt(self._mu * (1 - self._mu))


def log_bernoulli(x: Tensor, mu: Tensor) -> Tensor:
    """
    Ref:
        https://docs.chainer.org/en/stable/reference/generated/chainer.functions.bernoulli_nll.html
        https://github.com/emited/VariationalRecurrentNeuralNetwork/blob/0f23c87d11597ecf50ecbbf1dd37429861fd7aca/model.py#L181
    """

    return x * torch.log(mu + _eps) + (1 - x) * torch.log(1 - mu - _eps)
