import math
from numbers import Real
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import Self

from mypython.terminal import Color

from .. import config
from .base import Distribution, ProbParamsValueError, _eps, to_optional_tensor


class Normal(Distribution):
    r""""""

    """
    N(x | μ, σ) = N(x | cond_vars)
    """

    # torch.distributions.Normal

    ParamsReturnType = Tuple[Tensor, Tensor]

    def __init__(
        self,
        mu: Union[None, int, float, Tensor] = None,
        sigma: Union[None, int, float, Tensor] = None,
    ) -> None:
        super().__init__()

        self._mu_pvt_ = to_optional_tensor(mu)
        self._sigma_pvt_ = to_optional_tensor(sigma)

    def forward(self, *cond_vars, **cond_vars_k) -> ParamsReturnType:
        raise NotImplementedError()
        mu: Tensor
        sigma: Tensor
        return mu, sigma

    def __call__(self, *args, **kwargs) -> ParamsReturnType:
        return super().__call__(*args, **kwargs)

    def given(self, *cond_vars, **cond_vars_k) -> Self:
        self._cnt_given += 1 if self._cnt_given < 1024 else 0
        self._mu_pvt_, self._sigma_pvt_ = self(*cond_vars, **cond_vars_k)

        if config.check_value:
            assert type(self._mu_pvt_) == Tensor
            assert type(self._sigma_pvt_) == Tensor

            if self._mu_pvt_.isnan().any().item():
                raise ProbParamsValueError("μ contains nan (Normal)")
            if self._sigma_pvt_.isnan().any().item():
                raise ProbParamsValueError("Σ contains nan (Normal)")
            if not (0 <= self._sigma_pvt_).all().item():
                raise ProbParamsValueError("σ should be 0 ≤ σ (Normal)")

        return self

    def log_prob(self, x: Tensor) -> Tensor:
        assert self._mu_pvt_ is not None
        assert self._sigma_pvt_ is not None
        assert x.shape == self._mu_pvt_.shape

        if config.use_original:
            prob = self.func_log(x, self.loc, self.scale)
        else:
            # a little bit slow
            prob = torch.distributions.Normal(loc=self.loc, scale=self.scale).log_prob(x)

        # print(prob.shape)
        return prob

    def decode(self) -> Tensor:
        assert self._mu_pvt_ is not None
        assert self._sigma_pvt_ is not None
        return self._mu_pvt_.detach()

    def sample(self) -> Tensor:
        return self.rsample().detach()

    def rsample(self) -> Tensor:
        assert self._mu_pvt_ is not None
        assert self._sigma_pvt_ is not None

        if config.use_original:
            rsample = self.func_rsample(self.loc, self.scale)
        else:
            # a little bit slow
            rsample = torch.distributions.Normal(loc=self.loc, scale=self.scale).rsample()

        # print(rsample.shape)
        return rsample

    def dist_parameters(self) -> ParamsReturnType:
        assert self._mu_pvt_ is not None
        assert self._sigma_pvt_ is not None
        return self._mu_pvt_, self._sigma_pvt_

    def clear_dist_parameters(self) -> None:
        self._cnt_given = 0
        self._mu_pvt_ = None
        self._sigma_pvt_ = None

    @property
    def loc(self) -> Tensor:
        return self._mu_pvt_

    @property
    def scale(self) -> Tensor:
        return self._sigma_pvt_

    @property
    def param_mu(self) -> Tensor:
        return self._mu_pvt_

    @property
    def param_sigma(self) -> Tensor:
        return self._sigma_pvt_

    @staticmethod
    def func_log(x: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
        r"""Log-likelihood of a Gaussian distribution

        .. math::
            \begin{array}{ll} \\
                \log \mathcal{N}(x \mid \mu, \sigma^2) = - \frac{1}{2} \left( \exp \left( \frac{(x - \mu)^2}{\sigma^2} \right) + 2 \ln \sigma + \ln 2\pi \right)
            \end{array}

        Other References:
            https://github.com/Kaixhin/PlaNet/blob/28c8491bc01e8f1b911300749e04c308c03db051/main.py#L171
        """

        log_sigma = math.log(sigma) if isinstance(sigma, Real) else sigma.log()
        return -0.5 * (((x - mu) / sigma) ** 2 + 2 * log_sigma + math.log(2 * math.pi))

        # return -F.mse_loss(x, mu, reduction="none")
        # return -(x - mu) ** 2

    @staticmethod
    def func_rsample(mu: Tensor, sigma: Tensor) -> Tensor:
        """
        Reparametrization trick
        z = μ + σε where σ ~ N(0, 1)
        """

        # return eps.mul(sigma).add_(mu)
        eps = torch.randn_like(mu)
        return mu + sigma * eps

    @staticmethod
    def func_pdf(x: Tensor, mu: Tensor, sigma: Tensor):
        return torch.exp(-0.5 * (((x - mu) / sigma) ** 2)) / (math.sqrt(2.0 * math.pi) * sigma)


# Standard normal distribution
Normal01 = Normal(0, 1)
