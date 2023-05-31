import math
from numbers import Number, Real
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from typing_extensions import Self

from mypython.terminal import Color

from .. import config
from .base import Distribution, ProbParamsValueError, to_optional_tensor


class Bernoulli(Distribution):
    r""""""

    """
    Bern(x | μ) = Bern(x | cond_vars)
    """

    # torch.distributions.Bernoulli

    ParamsReturnType = Tensor

    def __init__(
        self,
        mu: Union[None, int, float, Tensor] = None,
    ) -> None:
        super().__init__()

        self._mu_pvt_ = to_optional_tensor(mu)

    def forward(self, *cond_vars: Tensor, **cond_vars_k: Tensor) -> ParamsReturnType:
        raise NotImplementedError()
        mu: Tensor
        return mu

    def __call__(self, *args, **kwargs) -> ParamsReturnType:
        return super().__call__(*args, **kwargs)

    def given(self, *cond_vars: Tensor, **cond_vars_k: Tensor) -> Self:
        self._cnt_given += 1 if self._cnt_given < 1024 else 0
        self._mu_pvt_ = self(*cond_vars, **cond_vars_k)

        if config.check_value:
            assert type(self._mu_pvt_) == Tensor

            if self._mu_pvt_.isnan().any().item():
                raise ProbParamsValueError("μ contains nan (Bernoulli)")
            if not ((0 <= self._mu_pvt_).all() and (self._mu_pvt_ <= 1).all()).item():
                raise ProbParamsValueError("μ should be 0 ≤ μ ≤ 1 (Bernoulli)")

        return self

    def log_prob(self, x: Tensor) -> Tensor:
        assert self._mu_pvt_ is not None
        assert x.shape == self._mu_pvt_.shape
        return self.func_log(x, self._mu_pvt_)

    def decode(self) -> Tensor:
        assert self._mu_pvt_ is not None
        return self._mu_pvt_.detach()

    def dist_parameters(self) -> ParamsReturnType:
        assert self._mu_pvt_ is not None
        return self._mu_pvt_

    def clear_dist_parameters(self) -> None:
        self._cnt_given = 0
        self._mu_pvt_ = None

    @property
    def loc(self) -> Tensor:
        return self._mu_pvt_

    @property
    def scale(self) -> Tensor:
        return torch.sqrt(self._mu_pvt_ * (1 - self._mu_pvt_))

    @property
    def param_mu(self) -> Tensor:
        return self._mu_pvt_

    @staticmethod
    def func_log(x: Tensor, mu: Tensor) -> Tensor:
        """
        References:
            https://docs.chainer.org/en/stable/reference/generated/chainer.functions.bernoulli_nll.html
            https://github.com/emited/VariationalRecurrentNeuralNetwork/blob/0f23c87d11597ecf50ecbbf1dd37429861fd7aca/model.py#L181
        """

        return x * torch.log(mu) + (1 - x) * torch.log(1 - mu)
