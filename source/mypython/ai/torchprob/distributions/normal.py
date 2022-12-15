from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from typing_extensions import Self

from mypython.terminal import Color

from .base import Distribution, _eps, _to_optional_tensor


class Normal(Distribution):
    # torch.distributions.Normal

    def __init__(
        self,
        mu: Union[None, int, float, Tensor] = None,
        sigma: Union[None, int, float, Tensor] = None,
    ) -> None:
        super().__init__()

        self._mu = _to_optional_tensor(mu)
        self._sigma = _to_optional_tensor(sigma)

    def forward(self, *cond_vars: Tensor, **cond_vars_k: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute μ, σ of N(x | μ, σ) from N(x | cond_vars)"""
        raise NotImplementedError()
        mu: Tensor
        sigma: Tensor
        return mu, sigma

    def cond(self, *cond_vars: Tensor, **cond_vars_k: Tensor) -> Self:
        self._mu, self._sigma = self(*cond_vars, **cond_vars_k)

        _check_sigma = self._sigma.nan_to_num()
        assert (0 <= _check_sigma).all().item(), "σ should be 0 <= σ (Normal distribution)"

        self._cnt += 1 if self._cnt < 1024 else 0
        return self

    def log_p(self, x: Tensor) -> Tensor:
        assert self._mu is not None
        assert self._sigma is not None
        assert x.shape == self._mu.shape
        return log_normal(x, self._mu, self._sigma)

    def decode(self) -> Tensor:
        assert self._mu is not None
        assert self._sigma is not None
        return self._mu.detach()

    def rsample(self) -> Tensor:
        assert self._mu is not None
        assert self._sigma is not None
        x = normal_sample(self._mu, self._sigma)
        return x

    def get_dist_params(self) -> Tuple[Tensor, Tensor]:
        assert self._mu is not None
        assert self._sigma is not None
        return self._mu, self._sigma

    @property
    def loc(self) -> Tensor:
        return self._mu

    @property
    def scale(self) -> Tensor:
        return self._sigma


# Standard normal distribution
Normal01 = Normal(0, 1)


def log_normal(x: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
    r"""Log-likelihood of a Gaussian distribution

    .. math::
        \begin{array}{ll} \\
            \log \mathcal{N}(x; \mu, \sigma^2) = - \frac{1}{2} \left( \exp \left( \frac{(x - \mu)^2}{\sigma^2} \right) + 2 \ln \sigma + \ln 2\pi \right)
        \end{array}

    If the sigma is fixed at 1, This equation is equivalent to :math:`- \frac{1}{2} (x - \mu)^2 + C`.
    This is equivalent to sum of squared errors (SSE) and is widely used as a mathematical expression for reconstruction error.
    That is, this :math:`\mu` is the image (, etc.) that is reconstructed.

    References:
        https://github.com/emited/VariationalRecurrentNeuralNetwork/blob/0f23c87d11597ecf50ecbbf1dd37429861fd7aca/model.py#L185
    """

    log_2pi = 0.79817986835  # log(2π)
    log_2pi_half = 0.39908993417  # log(2π)/2
    # sigma_p2 = sigma.pow(2)
    return -0.5 * (((x - mu) / sigma).pow(2) + 2 * torch.log(sigma + _eps) + log_2pi)


def normal_sample(mu: Tensor, sigma: Tensor) -> Tensor:
    """
    Reparametrization trick
    """

    # return eps.mul(sigma).add_(mu)
    eps = torch.randn_like(mu)
    return mu + eps * sigma
