from typing import Union

import torch
import torch.nn.functional as F
import torch.nn.init
from torch import Tensor, nn

import mypython.ai.torchprob as tp

from .cell import NewtonianVAECellFamily


class PurePControl:
    """
    Paper:
        we sample random starting and goal states, and successively
        apply the control law ut ∝ (x(Igoal) − xt(It)).
    """

    def __init__(
        self,
        Igoal: Tensor,
        alpha: float,
        cell: NewtonianVAECellFamily,
    ) -> None:
        self.alpha = alpha
        self.cell = cell
        self.x_goal = cell.q_encoder.cond(Igoal).rsample()

    def step(self, x_t: Tensor):
        u_t = self.alpha * (self.x_goal - x_t)
        return u_t


class PControl(tp.GMM):
    r"""
    Eq (12)

    .. math::
        \begin{array}{ll}
            P(\u_t \mid \x_t) = \displaystyle \sum_{n=1}^N \pi_n(\x_t) \mathcal{N}(\u_t \mid K_n(\x_n^{goal} - \x_t), \sigma_n^2)
        \end{array}
    """

    def __init__(self, N, dim_x, dim_x_goal_middle, dim_pi_middle, dim_K_middle, std=1.0) -> None:
        super().__init__()

        self.N = N
        self.x_g_W1 = nn.Parameter(torch.empty((N, dim_x_goal_middle)))
        self.x_g_W2 = nn.Parameter(torch.empty((dim_x_goal_middle, dim_x)))
        self.K_W1 = nn.Parameter(torch.empty((N, dim_K_middle)))
        self.K_W2 = nn.Parameter(torch.empty((dim_K_middle, dim_x)))
        self.pi_1 = nn.Linear(dim_x, dim_pi_middle)
        self.pi_2 = nn.Linear(dim_pi_middle, N)
        self.std = torch.tensor(std)

        # https://discuss.pytorch.org/t/somthing-about-linear-py-in-nn-moudles/109304
        torch.nn.init.kaiming_normal_(self.x_g_W1)
        torch.nn.init.kaiming_normal_(self.x_g_W2)
        torch.nn.init.kaiming_normal_(self.K_W1)
        torch.nn.init.kaiming_normal_(self.K_W2)

    def pi(self, x_t: Tensor) -> Tensor:
        """
        Returns:
            pi: (*, N)
        """
        pi = self.pi_1(x_t)
        pi = self.pi_2(pi)
        return pi

    def Kn(self) -> Tensor:
        """
        Returns:
            K_n: (N, D)
        """
        return self.K_W1 @ self.K_W2

    def x_goal_n(self) -> Tensor:
        return self.x_g_W1 @ self.x_g_W2

    def forward(self, x_t: Tensor):
        """
        x_t: shape (*, dim_x)
        """
        pi = torch.softmax(self.pi(x_t), dim=-1)  # (*, N)
        x_t = x_t.unsqueeze(-2)
        mu = self.Kn() * (self.x_goal_n() - x_t)  # (*, N, dim_x)
        return pi, mu, self.std.to(mu.device)

    def step(self, x_t: Tensor):
        u_t = self.cond(x_t).sample()
        return u_t
