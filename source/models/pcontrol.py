from typing import Union

import torch.nn as nn
from torch import Tensor

from .core import NewtonianVAECell, NewtonianVAEDerivationCell


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
        cell: Union[NewtonianVAECell, NewtonianVAEDerivationCell],
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
    """
    Paper:
        P(ut | xt) = N∑n=1 πn(xt) N(ut | Kn(xgoaln − xt), σ2n) (12)
    """

    def __init__(self) -> None:
        super().__init__()
