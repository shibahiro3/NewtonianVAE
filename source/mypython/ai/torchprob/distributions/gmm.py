import math
from numbers import Number, Real
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from typing_extensions import Self

from mypython.terminal import Color

from .. import config
from .base import Distribution, ProbParamsValueError, to_optional_tensor
from .normal import Normal


class GMM(Distribution):
    r"""
    Gaussian Mixture Model

    .. math::
        \begin{array}{ll}
            p(\boldsymbol{x} \mid \boldsymbol{\Theta}) = \displaystyle \sum_{k=1}^K \pi_k \, \mathcal{N}(\boldsymbol{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
        \end{array}

    "Normal (Gaussian) distribution" is not multivariate normal distribution.

    Constraint:

    .. math::
        \begin{array}{ll}
            \displaystyle \sum_{k=1}^K \pi_k = 1
        \end{array}

    Inputs: pi, mu, sigma
        * **pi**: tensor of shape :math:`(N, K)`
        * **mu**: tensor of shape :math:`(N, K, D)`
        * **sigma**: tensor of shape :math:`(N, K, D)`

        where:

        .. math::
            \begin{aligned}
                K ={} & \text{number of mixtures} \\
                D ={} & \text{number of features} \\
            \end{aligned}

    References:
        https://notebook.community/hardmaru/pytorch_notebooks/mixture_density_networks

    Other References:
        https://github.com/ctallec/world-models/blob/master/models/mdrnn.py
        https://github.com/sagelywizard/pytorch-mdn/blob/master/mdn/mdn.py

        http://www.allisone.co.jp/html/Notes/Mathematics/statistics/gmm/index.html#mjx-eqn-GMM
    """

    ParamsReturnType = Tuple[Tensor, Tensor, Tensor]

    def __init__(
        self,
        pi: Union[None, int, float, Tensor] = None,
        mu: Union[None, int, float, Tensor] = None,
        sigma: Union[None, int, float, Tensor] = None,
    ) -> None:
        super().__init__()

        self._pi_pvt_ = to_optional_tensor(pi)
        self._mu_pvt_ = to_optional_tensor(mu)
        self._sigma_pvt_ = to_optional_tensor(sigma)

    def forward(self, *cond_vars: Tensor, **cond_vars_k: Tensor) -> ParamsReturnType:
        raise NotImplementedError()
        pi: Tensor
        mu: Tensor
        sigma: Tensor
        return pi, mu, sigma

    def __call__(self, *args, **kwargs) -> ParamsReturnType:
        return super().__call__(*args, **kwargs)

    def given(self, *cond_vars: Tensor, **cond_vars_k: Tensor) -> Self:
        self._cnt_given += 1 if self._cnt_given < 1024 else 0
        self._pi_pvt_, self._mu_pvt_, self._sigma_pvt_ = self(*cond_vars, **cond_vars_k)

        if config.check_value:
            assert type(self._pi_pvt_) == Tensor
            assert type(self._mu_pvt_) == Tensor
            assert type(self._sigma_pvt_) == Tensor

            assert self._pi_pvt_.ndim == 2  # (N, K)
            assert self._mu_pvt_.ndim == 3  # (N, K, D)
            assert self._pi_pvt_.shape == self._mu_pvt_.shape[:-1]

            if self._sigma_pvt_.ndim == 3:  # (N, K, D)
                assert self._pi_pvt_.shape == self._sigma_pvt_.shape[:-1]
                assert self._mu_pvt_.shape[-1] == self._sigma_pvt_.shape[-1]
            elif self._sigma_pvt_.ndim == 0:
                pass
            else:
                assert False

            if self._pi_pvt_.isnan().any().item():
                raise ProbParamsValueError("π contains nan (GMM)")
            if self._mu_pvt_.isnan().any().item():
                raise ProbParamsValueError("μ contains nan (GMM)")
            if self._sigma_pvt_.isnan().any().item():
                raise ProbParamsValueError("Σ contains nan (GMM)")
            if not ((0 <= self._pi_pvt_).all() and (self._pi_pvt_ <= 1).all()).item():
                raise ProbParamsValueError("π should be 0 ≤ π ≤ 1 (GMM)")
            if not torch.isclose(self._pi_pvt_.sum(dim=-1), torch.tensor(1.0)).all().item():
                raise ProbParamsValueError("π should be Σ_k π_k = 1 (GMM)")
            if not (0 <= self._sigma_pvt_).all().item():
                raise ProbParamsValueError("Σ should be 0 ≤ diag(Σ) (GMM)")

        return self

    def log_prob(self, x: Tensor) -> Tensor:
        assert self._pi_pvt_ is not None
        assert self._mu_pvt_ is not None
        assert self._sigma_pvt_ is not None
        return self.func_log(x, self._pi_pvt_, self._mu_pvt_, self._sigma_pvt_)

    def sample(self):
        return self.rsample().detach()

    def rsample(self):
        whatever = True
        if whatever:
            rsample = self.func_rsample_gumbel(self._pi_pvt_, self._mu_pvt_, self._sigma_pvt_)
        else:
            rsample = self.func_rsample(self._pi_pvt_, self._mu_pvt_, self._sigma_pvt_)

        # print(rsample.shape)
        return rsample

    def dist_parameters(self) -> ParamsReturnType:
        return self._pi_pvt_, self._mu_pvt_, self._sigma_pvt_

    def clear_dist_parameters(self) -> None:
        self._cnt_given = 0
        self._pi_pvt_ = None
        self._mu_pvt_ = None
        self._sigma_pvt_ = None

    @property
    def param_pi(self):
        return self._pi_pvt_

    @property
    def param_mu(self):
        return self._mu_pvt_

    @property
    def param_sigma(self):
        return self._sigma_pvt_

    @staticmethod
    def func_log(x: Tensor, pi: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
        r"""
        Args:
            x: (N, D)
            pi: (N, K)
            mu: (N, K, D)
            sigma: (N, K, D) or scalar

        Returns:
            (N, D)

        .. math::
            \begin{array}{ll}
                \log p(\boldsymbol{X} \mid \boldsymbol{\pi}, \boldsymbol{\mu}, \boldsymbol{\Sigma})
                = \displaystyle \sum_{n=1}^N \log \sum_{k=1}^K \pi_k \, \mathcal{N}(\boldsymbol{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
            \end{array}

        Other References:
            https://github.com/ctallec/world-models/blob/master/models/mdrnn.py#L10
        """

        dist = GMM.func_pdf(x, mu, sigma)
        pi = pi.unsqueeze(-1)
        # assert (0 <= pi).all().item()
        # assert (0 <= dist).all().item()
        return (pi * dist).sum(-2).log()

    @staticmethod
    def func_rsample(pi: Tensor, mu: Tensor, sigma: Tensor):
        """
        Args:
            pi: (N, K)
            mu: (N, K, D)
            sigma: (N, K, D) or scalar

        Returns:
            (N, D)
        """

        indices = torch.distributions.Categorical(pi).sample()
        indices = indices.unsqueeze(0)
        N, _, D = mu.shape
        indices = indices.expand(D, N).T.unsqueeze(1)
        mu = mu.gather(1, indices).squeeze(1)

        # eps = torch.randn_like(mu)
        eps = torch.randn((N, D), device=mu.device)
        if sigma.ndim > 0:
            sigma = sigma.gather(1, indices).squeeze(1)

        return mu + sigma * eps

    @staticmethod
    def func_rsample_gumbel(pi: Tensor, mu: Tensor, sigma: Tensor):
        """
        Args:
            pi: (N, K)
            mu: (N, K, D)
            sigma: (N, K, D) or scalar

        Returns:
            (N, D)
        """

        z = torch.distributions.Gumbel(loc=0, scale=1).sample(pi.shape).to(mu.device)
        k = (torch.log(pi) + z).argmax(axis=-1)
        N, _, D = mu.shape
        indices = (torch.arange(N), k)

        eps = torch.randn((N, D), device=mu.device)
        if sigma.ndim == 0:
            return mu[indices] + sigma * eps
        else:
            return mu[indices] + sigma[indices] * eps

    @staticmethod
    def func_pdf(x: Tensor, mu: Tensor, sigma: Tensor):
        x = x.unsqueeze(-2)
        x = x.expand_as(mu)
        return Normal.func_pdf(x, mu, sigma)
