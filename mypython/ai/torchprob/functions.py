from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from scipy import special as sp
from torch import Tensor
from torch.distributions import kl

from .distributions.base import Distribution
from .distributions.normal import Normal

# from torch.distributions import kl_divergence
# from pixyz.losses.divergences import KullbackLeibler


def KLdiv(p: Distribution, q: Distribution) -> Tensor:
    if issubclass(type(p), Normal):
        if issubclass(type(q), Normal):
            return KL_normal_normal(p.loc, p.scale, q.loc, q.scale)
            # return kl._kl_normal_normal(p, q)

    else:
        assert False, "No KLD function"


def log(p: Distribution, x: Tensor, *cond_vars: Tensor) -> Tensor:
    """
    Match the look of the formula
    """
    return p.cond(*cond_vars).log_p(x)


def KL_normal_normal(
    mu_1: Tensor,
    sigma_1: Tensor,
    mu_2: Tensor,
    sigma_2: Tensor,
    eps=torch.finfo(torch.float).eps,
) -> Tensor:
    """
    Ref:
        https://github.com/emited/VariationalRecurrentNeuralNetwork/blob/0f23c87d11597ecf50ecbbf1dd37429861fd7aca/model.py#L172
    """
    if not (mu_1.shape == torch.Size([]) or mu_2.shape == torch.Size([])):
        assert mu_1.shape == mu_2.shape

    kld_element = (
        ((mu_1 - mu_2).pow(2) + sigma_1.pow(2)) / sigma_2.pow(2)
        + 2 * torch.log(sigma_2 + eps)
        - 2 * torch.log(sigma_1 + eps)
        - 1
    )
    return 0.5 * kld_element
