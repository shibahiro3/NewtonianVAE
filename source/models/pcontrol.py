from typing import Union

from torch import Tensor, nn

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

    def __init__(self) -> None:
        super().__init__()
