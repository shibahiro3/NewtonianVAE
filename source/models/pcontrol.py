from typing import Union

import torch
import torch.nn.functional as F
import torch.nn.init
from torch import Tensor, nn

import mypython.ai.torchprob as tp

from .core import NewtonianVAECell, NewtonianVAECellFamily, NewtonianVAEDerivationCell


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

    def get_action(self, I_t):
        x_t = self.cell.q_encoder.cond(I_t).rsample()
        return self.get_action_from_x(x_t)

    def get_action_from_x(self, x_t):
        u_t = self.alpha * (self.x_goal - x_t)
        return u_t


# import pixyz.distributions

# pixyz.distributions.MixtureModel
# pixyz.distributions.MixtureOfNormal


class PControl(nn.Module):
    r"""
    Eq (12)

    .. math::
        \begin{array}{ll}
            P(\u_t \mid \x_t) = \displaystyle \sum_{n=1}^N \pi_n(\x_t) \mathcal{N}(K_n(\x_{goal} - \x_t), \sigma^2)
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

    def pi(self, x_t) -> Tensor:
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
            Kn: (N, D)
        """
        return self.K_W1 @ self.K_W2

    def x_goal_n(self) -> Tensor:
        return self.x_g_W1 @ self.x_g_W2

    def forward(self, x_t):
        """
        x_t: shape (*, dim_x)
        """

        # l = self.pi(x_t).mean()
        # l.backward()

        log_pi = torch.log_softmax(self.pi(x_t), dim=-1)  # (*, N)

        # log_pi_ = torch.log_softmax(self.pi(x_t), dim=-1).mean()
        # log_pi_.backward()

        # Color.print(self.Kn().shape, c=Color.boldcyan)

        # 上でx_tを使ってるわ。だからtensorを書き換えてしまってはいけない。(でもrequires_grad=Falseやからええ気もするけど。　)
        # 上でx_t.clone()すれば良かったわ
        x_t = x_t.unsqueeze(-2)
        # print(x_t.requires_grad)
        # torch.tensor()
        # torch.unsqueeze
        mus = self.Kn() * (self.x_goal_n() - x_t)  # (*, N, dim_x)

        # mu_ = mu.mean()
        # mu_.backward()

        # Color.print(log_pi.shape, c=Color.red)
        # Color.print(mu.shape, c=Color.blue)

        return log_pi, mus, self.std
