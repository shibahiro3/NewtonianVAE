from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.distributions import Categorical
from torch.distributions.normal import Normal as torch_Normal
from typing_extensions import Self

from mypython.terminal import Color

from .base import Distribution, to_optional_tensor


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
        * **pi**: tensor of shape :math:`(*, K)`
        * **mu**: tensor of shape :math:`(*, K, D)`
        * **sigma**: tensor of shape :math:`(*, K, D)`

        where:

        .. math::
            \begin{aligned}
                K ={} & \text{number of mixtures} \\
                D ={} & \text{number of features} \\
            \end{aligned}

    References:
        https://github.com/ctallec/world-models/blob/master/models/mdrnn.py
        https://github.com/sagelywizard/pytorch-mdn/blob/master/mdn/mdn.py

        http://www.allisone.co.jp/html/Notes/Mathematics/statistics/gmm/index.html#mjx-eqn-GMM
    """

    def __init__(
        self,
        pi: Union[None, int, float, Tensor] = None,
        mu: Union[None, int, float, Tensor] = None,
        sigma: Union[None, int, float, Tensor] = None,
    ) -> None:
        super().__init__()

        self._pi = to_optional_tensor(pi)
        self._mu = to_optional_tensor(mu)
        self._sigma = to_optional_tensor(sigma)

    def forward(self, *cond_vars: Tensor, **cond_vars_k: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError()
        pi: Tensor
        mu: Tensor
        sigma: Tensor
        return pi, mu, sigma

    def cond(self, *cond_vars: Tensor, **cond_vars_k: Tensor) -> Self:
        self._pi, self._mu, self._sigma = self(*cond_vars, **cond_vars_k)

        assert torch.isclose(self._pi.sum(dim=-1), torch.tensor(1.0)).all()
        assert (
            (0 <= self._sigma.nan_to_num()).all().item()
        ), "Σ should be 0 <= diag(Σ) (GMM distribution)"

        self._cnt += 1 if self._cnt < 1024 else 0
        return self

    def log_prob(self, x: Tensor) -> Tensor:
        assert self._pi is not None
        assert self._mu is not None
        assert self._sigma is not None
        return log_gmm(x, self._pi, self._mu, self._sigma)

    def sample(self):
        return self.rsample().detach()

    def rsample(self):
        return sample_gmm(self._pi, self._mu, self._sigma)

    def dist_parameters(self):
        return self._pi, self._mu, self._sigma


def log_gmm(x: Tensor, pi: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
    r"""
    .. math::
        \begin{array}{ll}
            \log p(\boldsymbol{X} \mid \boldsymbol{\pi}, \boldsymbol{\mu}, \boldsymbol{\Sigma})
              = \displaystyle \sum_{n=1}^N \log \sum_{k=1}^K \pi_k \, \mathcal{N}(\boldsymbol{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
        \end{array}

    References:
        https://github.com/ctallec/world-models/blob/master/models/mdrnn.py#L10
    """

    x = x.unsqueeze(-2)
    normal_dist = torch_Normal(mu, sigma)
    log_normals = normal_dist.log_prob(x)
    return (pi.log().unsqueeze(-1) + log_normals).sum(-2)  # sum : dim K


def sample_gmm(pi: Tensor, mu: Tensor, sigma: Tensor):
    """
    Args:
        pi: (N, K)
        mu: (N, K, D)
        sigma: (N, K, D)

    Returns:
        (N, D)
    """

    indices = Categorical(pi).sample()
    indices = indices.repeat(mu.shape[-1], 1).T.unsqueeze(1)
    mu = mu.gather(1, indices).squeeze(1)
    eps = torch.randn_like(mu)
    sigma = sigma.gather(1, indices).squeeze(1)
    return mu + eps * sigma


# import pixyz.distributions
# pixyz.distributions.MixtureModel
# pixyz.distributions.MixtureOfNormal
