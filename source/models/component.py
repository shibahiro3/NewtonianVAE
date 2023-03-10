r"""
.. math::
    \newcommand{\I}{\mathbf{I}}
    \newcommand{\u}{\mathbf{u}}
    \newcommand{\v}{\mathbf{v}}
    \newcommand{\x}{\mathbf{x}}
    \newcommand{\xhat}{\hat{\mathbf{x}}}
    \newcommand{\KL}[2]{\operatorname{KL}\left[ #1 \, \middle\| \, #2 \right]}
"""

from numbers import Real
from pprint import pprint
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import NumberType, Tensor, nn

import mypython.ai.torchprob as tp
from mypython.ai.util import find_function, swap01
from mypython.terminal import Color

from . import parts


class ABCf(nn.Module):
    r"""

    .. math::
        \begin{array}{ll} \\
            A = \mathrm{diag}(f(\x_{t-1}, \v_{t-1}, \u_{t-1})) \\
            B = -\log \mathrm{diag}(f(\x_{t-1}, \v_{t-1}, \u_{t-1})) \\
            C = \log \mathrm{diag}(f(\x_{t-1}, \v_{t-1}, \u_{t-1}))
        \end{array}


    Outputs: diagA, diagB, diagC
        * **diagA**: tensor of shape :math:`(*, D)`
        * **diagB**: tensor of shape :math:`(*, D)`
        * **diagC**: tensor of shape :math:`(*, D)`

        where:

        .. math::
            \begin{aligned}
                D ={} & \mathrm{dim}(\u) \\
            \end{aligned}


    References in paper:
        * [A, log (-B), log C] = diag(f(xt, vt, ut)),
        where f is a neural network with linear output activation.

        * To compute the transition matrices as a function of the state we use a fully
        connected network with 2 hidden layers with 16 units
        and ReLU activation, with the appropriate input and
        output dimensionality.

    References in paper (TS-NVAE):
        A  = diag(fA(xt, vt, ut))

        log (-B) = diag(fB(xt, vt, ut))

        log C    = diag(fC(xt, vt, ut))
    """

    def __init__(
        self,
        dim_x: int,
        activation: str = "ReLU",
    ) -> None:
        super().__init__()

        Activation = getattr(nn, activation)

        self.func = nn.Sequential(
            nn.Linear(dim_x * 3, dim_x),
            Activation(),
            # === 2 hidden layer
            nn.Linear(dim_x, dim_x),
            Activation(),
            nn.Linear(dim_x, dim_x),
            Activation(),
            # ===
            nn.Linear(dim_x, dim_x * 3),
        )

    def forward(self, x_tn1: Tensor, v_tn1: Tensor, u_tn1: Tensor):
        """"""
        abc = self.func(torch.cat([x_tn1, v_tn1, u_tn1], dim=-1))
        diagA, n_log_diagB, log_diagC = torch.chunk(abc, 3, dim=-1)
        return diagA, -n_log_diagB.exp(), log_diagC.exp()


class Velocity(nn.Module):
    r"""
    .. math::
        \begin{array}{ll}
            \v_t = \v_{t-1} + \Delta t \cdot (A\x_{t-1} + B\v_{t-1} + C\u_{t-1}) \\
        \end{array}
    """

    def __init__(
        self,
        *,
        dim_x: int,
        fix_abc: Union[None, Tuple[NumberType, NumberType, NumberType]] = None,
        activation: str = "ReLU",
    ) -> None:
        super().__init__()
        self.dim_x = dim_x

        if fix_abc is None:
            self.func_abc = ABCf(dim_x, activation)
        else:
            assert len(fix_abc) == 3
            self.func_abc = None

        self.fix_abc = fix_abc

    def forward(self, x_tn1: Tensor, u_tn1: Tensor, v_tn1: Tensor, dt: Tensor):
        """"""

        """
        if A is diagonal matrix:
            A @ x == diag(A) * x
        """

        if self.fix_abc is None:
            diagA, diagB, diagC = self.func_abc(x_tn1, u_tn1, v_tn1)
        else:
            # Paper:
            #   unbounded and full rank
            #   Firstly, the transition matrices were set to A = 0, B = 0, C = 1.
            diagA, diagB, diagC = self.fix_abc

        # Color.print(diagA)
        # Color.print(diagB, c=Color.red)
        # Color.print(diagC, c=Color.blue)

        v_t = v_tn1 + dt * (diagA * x_tn1 + diagB * v_tn1 + diagC * u_tn1)
        return v_t

    def __call__(self, *args, **kwargs) -> Tensor:
        return super().__call__(*args, **kwargs)


class Transition(tp.Normal):
    r"""Transition prior. Eq (9).

    .. math::
        \begin{array}{ll}
            p(\x_t \mid \x_{t-1}, \u_{t-1}) = \mathcal{N}(\x_t \mid \x_{t-1} + \Delta t \cdot \v_t, \sigma^2) \\
        \end{array}
    """

    def __init__(self, std: Optional[Real]) -> None:
        super().__init__()

        if std is not None:
            self.std: Tensor
            self.register_buffer("std", torch.tensor(std))
            self.std_function = lambda x: x
        else:
            # σ is not σ_t, so σ should not depend on x_t.
            self.std = nn.Parameter(torch.randn(1))
            self.std_function = F.softplus

    def forward(self, x_tn1: Tensor, v_t: Tensor, dt: Tensor):
        x_t = x_tn1 + dt * v_t
        mu = x_t
        sigma = self.std_function(self.std)
        # print(sigma.item())
        return mu, sigma


class Encoder(tp.Normal):
    r"""
    .. math::
        \begin{array}{ll}
            q(\x_t \mid \I_t)
        \end{array}

    References in paper:
        We use Gaussian p(It | xt) and q(xt | It) parametrized by a neural network throughout.
    """

    def __init__(
        self,
        *,
        dim_x: int,
        dim_output: int,
        model: str,
        model_kwargs={},
        std_function: str = "softplus",
    ) -> None:
        super().__init__()

        self.enc = getattr(parts, model)(dim_output=dim_output, **model_kwargs)
        self.mean_std = nn.Linear(dim_output, dim_x * 2)

        # nn.Exp() does't exist
        self.std_function = find_function(std_function)

    def forward(self, I_t: Tensor):
        """"""
        middle = self.enc(I_t)
        middle = self.mean_std(middle)
        mu, sigma = torch.chunk(middle, 2, dim=-1)
        sigma = self.std_function(sigma)
        # sigma += torch.finfo(sigma.dtype).eps
        return mu, sigma


class Decoder(nn.Module):
    r"""
    .. math::
        \begin{array}{ll}
            p(\I_t \mid \x_{t}) \hspace{5mm} \text{or} \hspace{5mm} p(\I_t \mid \xhat_{t})
        \end{array}

    obsolete : std

    References in paper:
        We use Gaussian p(It | xt) and q(xt | It) parametrized by a neural network throughout.
    """

    def __init__(
        self,
        *,
        dim_input: int,
        model: str,
        model_kwargs={},
    ) -> None:
        super().__init__()

        self.dec = getattr(parts, model)(dim_input=dim_input, **model_kwargs)

    def forward(self, x_t: Tensor):
        """"""
        out = self.dec(x_t)
        return out


class Pxhat(tp.Normal):
    r"""
    .. math::
        \begin{array}{ll}
            p(\xhat_t \mid \x_{t-1}, \u_{t-1})
        \end{array}

    It is not clearly defined in the original paper.
    """

    def __init__(
        self,
        dim_x: int,
        dim_xhat: int,
        dim_middle: int,
        std_function: str = "softplus",
    ) -> None:
        super().__init__()

        self.fc = nn.Linear(2 * dim_x, dim_middle)
        self.mean = nn.Linear(dim_middle, dim_xhat)
        self.std = nn.Linear(dim_middle, dim_xhat)

        # nn.Exp() does't exist
        self.std_function = find_function(std_function)

    def forward(self, x_tn1: Tensor, u_tn1: Tensor):
        middle = torch.cat([x_tn1, u_tn1], dim=-1)
        middle = self.fc(middle)
        mu = self.mean(middle)
        sigma = self.std_function(self.std(middle))
        # sigma += torch.finfo(sigma.dtype).eps
        return mu, sigma


class MultiEncoder(tp.Normal):
    def __init__(
        self,
        dim_x: int,
        modellist: dict,
        std_function: str = "softplus",
    ) -> None:
        super().__init__()

        self.dim_x = dim_x

        sum_dim_output = 0
        self.encoders = nn.ModuleList()
        for m in modellist:
            model_ = getattr(parts, m["model"])(**m["model_kwargs"])
            self.encoders.append(model_)
            sum_dim_output += m["model_kwargs"]["dim_output"]

        self.mish = nn.Mish()
        self.mean_std = nn.Linear(sum_dim_output, dim_x * 2)

        # nn.Exp() does't exist
        self.std_function = find_function(std_function)

    def forward(self, I_t: List[Tensor]):
        """"""

        middles = []
        for i, enc in enumerate(self.encoders):
            middles.append(enc(I_t[i]))

        middle = torch.cat(middles, dim=-1)
        middle = self.mish(middle)
        mean_std = self.mean_std(middle)
        mu, sigma = torch.chunk(mean_std, 2, dim=-1)
        sigma = self.std_function(sigma)
        # sigma += torch.finfo(sigma.dtype).eps
        return mu, sigma


class MultiDecoder(nn.Module):
    """
    obsolete : std
    """

    def __init__(
        self,
        dim_x: int,
        split_version: str,
        modellist: dict,
    ) -> None:
        super().__init__()
        assert split_version in ("v1", "none")

        self.dim_x = dim_x

        self.decoders = nn.ModuleList()
        self.dim_inputs = []
        for m in modellist:
            model_ = getattr(parts, m["model"])(**m["model_kwargs"])
            self.decoders.append(model_)
            self.dim_inputs.append(m["model_kwargs"]["dim_input"])

        self.split = split_version
        if split_version == "v1":
            self.mish = nn.Mish()
            self.fc = nn.Linear(dim_x, sum(self.dim_inputs))

    def __call__(self, *args, **kwargs) -> List[Tensor]:
        return super().__call__(*args, **kwargs)

    def forward(self, x_t: Tensor):
        """"""

        outputs = []

        if self.split == "v1":
            middle = self.fc(x_t)
            middle = self.mish(middle)
            start = 0
            for i, dec in enumerate(self.decoders):
                span = self.dim_inputs[i]
                outputs.append(dec(middle[..., start : start + span]))
                start += span

        elif self.split == "none":
            for dec in self.decoders:
                outputs.append(dec(x_t))

        return outputs
