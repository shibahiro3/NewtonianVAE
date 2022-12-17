r"""
.. math::
    \newcommand{\I}{\mathbf{I}}
    \newcommand{\u}{\mathbf{u}}
    \newcommand{\v}{\mathbf{v}}
    \newcommand{\x}{\mathbf{x}}
    \newcommand{\xhat}{\hat{\mathbf{x}}}
    \newcommand{\KL}[2]{\operatorname{KL}\left[ #1 \, \middle\| \, #2 \right]}
"""

from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
from torch import NumberType, Tensor, nn

import mypython.ai.torchprob as tp
from mypython.terminal import Color


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
        [A, log (-B), log C] = diag(f(xt, vt, ut)),
        where f is a neural network with linear output activation.

    References in paper (TS-NVAE):
        A  = diag(fA(xt, vt, ut))

        log (-B) = diag(fB(xt, vt, ut))

        log C    = diag(fC(xt, vt, ut))
    """

    def __init__(
        self,
        dim_x: int,
    ) -> None:
        super().__init__()

        dim_IO = 3 * dim_x

        self.func = nn.Sequential(
            nn.Linear(dim_IO, dim_IO),
            nn.Softplus(),
            nn.Linear(dim_IO, dim_IO),
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
        dim_x: int,
        fix_abc: Union[None, Tuple[NumberType, NumberType, NumberType]] = None,
    ) -> None:
        super().__init__()

        if fix_abc is None:
            self.func_abc = ABCf(dim_x)
        else:
            assert len(fix_abc) == 3
            self.func_abc = None

        self.fix_abc = fix_abc

    def forward(self, x_tn1: Tensor, u_tn1: Tensor, v_tn1: Tensor, dt: float):
        """"""

        """
        if A is diagonal matrix:
            A @ x == diag(A) * x
        """

        if self.fix_abc is None:
            diagA, diagB, diagC = self.func_abc(x_tn1, u_tn1, v_tn1)
            # diagA, diagB が 1、すなわちxとvにかかっていたら、簡単にnanになる
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

    def __init__(self, std=1.0) -> None:
        super().__init__()

        self.std = torch.tensor(std)

    def forward(self, x_tn1: Tensor, v_t: Tensor, dt: float):
        x_t = x_tn1 + dt * v_t
        return x_t, self.std


class Encoder(tp.Normal):
    r"""
    .. math::
        \begin{array}{ll}
            q(\x_t \mid \I_t)
        \end{array}

    References in paper:
        We use Gaussian p(It | xt) and q(xt | It) parametrized by a neural network throughout.
    """

    def __init__(self, dim_x: int, dim_middle: int, std_function: Callable) -> None:
        super().__init__()

        self.fc = VisualEncoder64(dim_output=dim_middle)
        self.mean = nn.Linear(dim_middle, dim_x)
        self.std = nn.Linear(dim_middle, dim_x)
        # self.std = torch.tensor(1)
        self.std_function = std_function

    def forward(self, I_t: Tensor):
        """"""
        middle = self.fc(I_t)
        mu = self.mean(middle)
        sigma = self.std_function(self.std(middle))
        # std = self.std
        return mu, sigma + torch.finfo(sigma.dtype).eps


class Decoder(tp.Normal):
    r"""
    .. math::
        \begin{array}{ll}
            p(\I_t \mid \x_{t}) \hspace{5mm} \text{or} \hspace{5mm} p(\I_t \mid \xhat_{t})
        \end{array}

    References in paper:
        We use Gaussian p(It | xt) and q(xt | It) parametrized by a neural network throughout.
    """

    def __init__(self, dim_x: int, decoder_type, std=1.0) -> None:
        super().__init__()

        if decoder_type == "VisualDecoder64":
            self.dec = VisualDecoder64(dim_x, 1024)
        elif decoder_type == "SpatialBroadcastDecoder64":
            self.dec = SpatialBroadcastDecoder64(dim_x)
        else:
            assert False

        self.std = torch.tensor(std)

    def forward(self, x_t: Tensor):
        """"""
        return self.dec(x_t), self.std + torch.finfo(self.std.dtype).eps


class Pxhat(tp.Normal):
    r"""
    .. math::
        \begin{array}{ll}
            p(\xhat_t \mid \x_{t-1}, \u_{t-1})
        \end{array}

    paperに何の説明もない
    """

    def __init__(self, dim_x: int, dim_xhat: int, dim_middle: int, std_function: Callable) -> None:
        super().__init__()

        self.fc = nn.Linear(2 * dim_x, dim_middle)
        self.mean = nn.Linear(dim_middle, dim_xhat)
        self.std = nn.Linear(dim_middle, dim_xhat)
        self.std_function = std_function

    def forward(self, x_tn1: Tensor, u_tn1: Tensor):
        middle = torch.cat([x_tn1, u_tn1], dim=-1)
        middle = self.fc(middle)
        mu = self.mean(middle)
        sigma = self.std_function(self.std(middle))
        return mu, sigma + torch.finfo(sigma.dtype).eps


class VisualEncoder64(nn.Module):
    r"""
    Inputs: x
        * **x**: tensor of shape :math:`(*, 3, 64, 64)`

    Outputs: y
        * **y**: tensor of shape :math:`(*, \mathrm{dim\_output})`

    Reference for implementation:
        https://github.com/ctallec/world-models/blob/master/models/vae.py#L32
    """

    def __init__(self, dim_output: int, activation=torch.relu) -> None:
        super().__init__()

        self.activation = activation
        self.dim_output = dim_output

        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        self.fc = nn.Identity() if dim_output == 1024 else nn.Linear(1024, dim_output)

    def forward(self, x: Tensor):
        """"""
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        x = x.reshape(-1, 1024)
        # Identity if embedding size is 1024 else linear projection
        y = self.fc(x)
        return y


class VisualDecoder64(nn.Module):
    r"""
    Inputs: z
        * **x**: tensor of shape :math:`(*, \mathrm{dim\_input})`

    Outputs: y
        * **y**: tensor of shape :math:`(*, 3, 64, 64)`

    Reference for implementation:
        https://github.com/ctallec/world-models/blob/master/models/vae.py#L10
    """

    def __init__(self, dim_input: int, dim_middle: int, activation=torch.relu):
        super().__init__()

        self.activation = activation
        self.dim_middle = dim_middle

        self.fc1 = nn.Linear(dim_input, dim_middle)
        self.conv1 = nn.ConvTranspose2d(dim_middle, 128, 5, stride=2)
        self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.conv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)

    def forward(self, z: Tensor):
        """"""
        z = self.fc1(z)
        z = z.reshape(-1, self.dim_middle, 1, 1)
        z = self.activation(self.conv1(z))
        z = self.activation(self.conv2(z))
        z = self.activation(self.conv3(z))
        y = self.conv4(z)
        return y


class SpatialBroadcastDecoder64(nn.Module):
    """
    Inputs: z
        * **x**: tensor of shape :math:`(N, \mathrm{dim\_input})`

    Outputs: y
        * **y**: tensor of shape :math:`(N, 3, 64, 64)`

    Reference:
        Spatial Broadcast Decoder https://arxiv.org/abs/1901.07017

    Reference for implementation:
        https://github.com/dfdazac/vaesbd/blob/master/model.py#L6
    """

    def __init__(self, dim_input: int) -> None:
        super().__init__()

        a = np.linspace(-1, 1, 64)
        b = np.linspace(-1, 1, 64)
        x_grid, y_grid = np.meshgrid(a, b)
        x_grid = torch.from_numpy(x_grid)
        y_grid = torch.from_numpy(y_grid)
        # Add as constant, with extra dims for N and C
        self.x_grid = nn.Parameter(x_grid.reshape((1, 1) + x_grid.shape), requires_grad=False)
        self.y_grid = nn.Parameter(y_grid.reshape((1, 1) + y_grid.shape), requires_grad=False)

        self.dec_convs = nn.Sequential(
            nn.Conv2d(dim_input + 2, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
        )

    def forward(self, z: Tensor):
        N = z.size(0)
        # View z as 4D tensor to be tiled across new H and W dimensions
        z = z.reshape(z.shape + (1, 1))  # (N, D, 1, 1)

        # Tile across to match image size
        z = z.expand(-1, -1, 64, 64)  # (N, D, 64, 64)

        # Expand grids to batches and concatenate on the channel dimension
        z = torch.cat(
            [
                self.x_grid.expand(N, -1, -1, -1),
                self.y_grid.expand(N, -1, -1, -1),
                z,
            ],
            dim=1,
        )  # (N, D+2, 64, 64)

        y = self.dec_convs(z)
        return y
