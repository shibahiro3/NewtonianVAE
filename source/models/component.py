import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torch import Tensor

import mypython.ai.torchprob as tp


class ABCf(nn.Module):
    """
    Paper:
        [A, log (-B), log C] = diag(f(xt, vt, ut)),
        where f is a neural network with linear output activation.

    Paper (TS-NVAE):
              A  = diag(fA(xt, vt, ut))
        log (-B) = diag(fB(xt, vt, ut))
        log C    = diag(fC(xt, vt, ut))
    """

    def __init__(self, dim_x: int) -> None:
        super().__init__()

        self.f = nn.Linear(3 * dim_x, 3 * dim_x)

    def forward(self, x_tn1: Tensor, v_tn1: Tensor, u_tn1: Tensor):
        """
        Returns:
            diag(A), diag(B), diag(C)
        """
        abc = self.f(torch.cat([x_tn1, v_tn1, u_tn1], dim=1))
        diagA, log_diagnB, log_diagC = torch.chunk(abc, 3, dim=1)
        return diagA, -log_diagnB.exp(), log_diagC.exp()


class Velocity(nn.Module):
    def __init__(self, dim_x: int) -> None:
        super().__init__()

        self.f_abc = ABCf(dim_x)

    def forward(self, x_tn1: Tensor, u_tn1: Tensor, v_tn1: Tensor, dt: float):
        """
        if A is diagonal matrix:
            A @ x == diag(A) * x
        """

        diagA, diagB, diagC = self.f_abc(x_tn1, u_tn1, v_tn1)
        # diagA, diagB が 1、すなわちxとvにかかっていたら、簡単にnanになる

        # unbounded and full rank
        # Firstly, the transition matrices were set to A = 0, B = 0, C = 1.
        diagA = torch.tensor(0)
        diagB = torch.tensor(0)
        diagC = torch.tensor(1)

        # Color.print(diagA)
        # Color.print(diagB, c=Color.red)
        # Color.print(diagC, c=Color.blue)

        v_t = v_tn1 + dt * (diagA * x_tn1 + diagB * v_tn1 + diagC * u_tn1)
        return v_t


class Transition(tp.Normal):
    """
    transition prior
    p(xt | x_{t-1}, u_{t-1}; vt)

    Paper:
        p(xt | x_{t-1}, u_{t-1}; vt) =  N(xt | x_{t-1} + ∆t·vt, σ^2)   (9)
    """

    def __init__(self, std=1.0) -> None:
        super().__init__()

        self.std = torch.tensor(std)

    def forward(self, x_tn1: Tensor, v_t: Tensor, dt: float):
        mu = x_tn1 + dt * v_t
        return mu, self.std


class Encoder(tp.Normal):
    """
    q(x_t | I_t)

    Paper:
        We use Gaussian p(It | xt) and q(xt | It) parametrized by a neural network throughout.
    """

    def __init__(self, dim_x: int, dim_middle=512) -> None:
        super().__init__()

        self.fc = VisualEncoder64(dim_output=dim_middle)
        self.mean = nn.Linear(dim_middle, dim_x)
        self.std = nn.Linear(dim_middle, dim_x)
        # self.std = torch.tensor(1)

    def forward(self, I_t: Tensor):
        middle = self.fc(I_t)
        mu = self.mean(middle)
        sigma = torch.relu(self.std(middle))
        # std = self.std
        return mu, sigma


class Decoder(tp.Normal):
    """
    p(I_t | x_t)
      or
    p(I_t | xhat_t)
    """

    def __init__(self, dim_x: int, std=1.0) -> None:
        super().__init__()

        self.dec = VisualDecoder64(dim_x, 1024)
        self.std = torch.tensor(std)

    def forward(self, x_t: Tensor):
        return self.dec(x_t), self.std


class Pxhat(tp.Normal):
    """
    p(xhat_t | x_{t-1}, u_{t-1})

    paperに何の説明もない
    """

    def __init__(self, dim_x: int, dim_xhat: int, dim_middle: int) -> None:
        super().__init__()

        # Color.print(dim_x)
        self.fc = nn.Linear(2 * dim_x, dim_middle)
        self.mean = nn.Linear(dim_middle, dim_xhat)
        self.std = nn.Linear(dim_middle, dim_xhat)

    def forward(self, x_tn1: Tensor, u_tn1: Tensor):
        middle = torch.cat([x_tn1, u_tn1], dim=1)
        # Color.print(middle.shape)
        middle = self.fc(middle)
        mu = self.mean(middle)
        sigma = self.std(middle).exp()
        return mu, sigma


class VisualEncoder64(nn.Module):
    """
    (*, 3, 64, 64) -> (*, dim_output)
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
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        x = x.reshape(-1, 1024)
        # Identity if embedding size is 1024 else linear projection
        x = self.fc(x)
        return x


class VisualDecoder64(nn.Module):
    """
    (*, dim_input) -> (*, 3, 64, 64)
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

    def forward(self, x: Tensor):
        x = self.fc1(x)
        x = x.reshape(-1, self.dim_middle, 1, 1)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.conv4(x)
        return x
