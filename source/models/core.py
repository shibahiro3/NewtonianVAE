"""
x_t    == x_t
x_tn1  == x_{t-1}

TODO:
    Paper:
    In the point mass experiments
    we found it useful to anneal the KL term in the ELBO,
    starting with a value of 0.001 and increasing it linearly
    to 1.0 between epochs 30 and 60.
"""

from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torch import Tensor

import mypython.ai.torchprob as tp

from .component import Decoder, DecoderDerivation, Encoder, Pxhat, Transition, Velocity


class NewtonianVAECellBase(nn.Module):
    def __init__(
        self,
        dim_x: int,
        transition_std: float,
        regularization: bool,
    ):
        super().__init__()
        self._dim_x = dim_x

        # v_t = v_{t-1} + Δt・(Ax_{t-1} + Bv_{t-1} + Cu_{t-1})
        self.f_velocity = Velocity(dim_x)

        # p(x_t | x_{t-1}, u_{t-1}; v_t)
        self.p_transition = Transition(transition_std)

        # q(x_t | I_t) (posterior)
        self.q_encoder = Encoder(dim_x)

        self.regularization = regularization

    @property
    def dim_x(self):
        return self._dim_x


class NewtonianVAECell(NewtonianVAECellBase):
    """
    Eq (11) : t -> t-1, +KL -> -KL (mistake)

    Ignore v_t or v_{t-1}

    L = E_{q(x_{t-1} | I_{t-1}) q(x_{t-2} | I_{t-2})}
            [ E_{p(x_t | x_{t-1}, u_{t-1}; v_{t-1})
               [ log p(I_t | x_t) - KL(q(x_t | I_t)‖ p(x_t | x_{t-1}, u_{t-1}; v_{t-1})) ] ]
    """

    def __init__(
        self,
        dim_x: int,
        transition_std: float,
        regularization: bool,
    ) -> None:
        super().__init__(dim_x, transition_std, regularization)

        # p(I_t | x_t)
        self.p_decoder = Decoder(dim_x)

    def forward(self, I_t: Tensor, x_tn1: Tensor, u_tn1: Tensor, v_tn1: Tensor, dt: float):

        if self.training:
            v_t = self.f_velocity(x_tn1, u_tn1, v_tn1, dt)

            self.q_encoder.cond(I_t)
            self.p_transition.cond(x_tn1, v_t, dt)

            # FIXME: From which distribution should we sample?
            x_t = self.q_encoder.rsample()
            # x_t = self.p_transition.rsample()

            # *_t について、各次元(ピクセル)についてはsum, batch size については mean
            E_ll = tp.log(self.p_decoder, I_t, x_t).sum(dim=(1, 2, 3)).mean(dim=0)
            E_kl = tp.KLdiv(self.q_encoder, self.p_transition).sum(dim=1).mean(dim=0)

            E = E_ll - E_kl

            if self.regularization:
                # Paper:
                # we added an additional regularization term
                # to the latent space, KL(q(x|I)‖N (0, 1))
                E -= tp.KLdiv(self.q_encoder, tp.Normal01).sum(dim=1).mean(dim=0)

            # FIXME: From which distribution should we sample?
            # x_t = self.p_transition.rsample()
            # x_t = self.q_encoder.rsample()

            return E, E_ll.detach(), E_kl.detach(), x_t, v_t

        else:
            # Paper:
            # During inference, vt is computed as
            # vt = (xt − xt−1)/∆t, with xt ∼ q(xt|It) and xt−1 ∼ q(xt−1|It−1).
            x_t = self.q_encoder.cond(I_t).rsample()
            v_t = (x_t - x_tn1) / dt

            # for decode
            self.p_decoder.cond(x_t)

            return 0, 0, 0, x_t, v_t


class NewtonianVAEDerivationCell(NewtonianVAECellBase):
    """
    Eq (23)
    """

    def __init__(
        self,
        dim_x: int,
        dim_xhat: int,
        dim_pxhat_middle: int,
        transition_std: float,
        regularization: bool,
    ) -> None:
        super().__init__(dim_x, transition_std, regularization)
        self._dim_xhat = dim_xhat

        # p(I_t | xhat_t)
        self.p_decoder = DecoderDerivation(dim_xhat)

        # p(xhat_t | x_{t-1}, u_{t-1})
        self.p_xhat = Pxhat(dim_x, dim_xhat, dim_pxhat_middle)

    def forward(self, I_t: Tensor, x_tn1: Tensor, u_tn1: Tensor, v_tn1: Tensor, dt: float):

        if self.training:
            v_t = self.f_velocity(x_tn1, u_tn1, v_tn1, dt)

            # E_{p(xhat_t | x_{t-1}, u_{t-1})}[...]
            xhat_t = self.p_xhat.cond(x_tn1, u_tn1).rsample()

            self.p_transition.cond(x_tn1, v_t, dt)
            self.q_encoder.cond(I_t)

            E_ll = tp.log(self.p_decoder, I_t, xhat_t).sum(dim=(1, 2, 3)).mean(dim=0)
            E_kl = tp.KLdiv(self.q_encoder, self.p_transition).sum(dim=1).mean(dim=0)

            E = E_ll - E_kl

            if self.regularization:
                # Paper:
                # we added an additional regularization term
                # to the latent space, KL(q(x|I)‖N (0, 1))
                E -= tp.KLdiv(self.q_encoder, tp.Normal01).sum(dim=1).mean(dim=0)

            # E_{q(x_{t-1} | I_{t-1}) ...}[ E[...] ]  (for next step: x_t of code becomes x_tn1)
            x_t = self.q_encoder.rsample()

            return E, E_ll.detach(), E_kl.detach(), x_t, v_t

        else:
            # Paper:
            # During inference, vt is computed as
            # vt = (xt − xt−1)/∆t, with xt ∼ q(xt|It) and xt−1 ∼ q(xt−1|It−1).
            x_t = self.q_encoder.cond(I_t).rsample()
            v_t = (x_t - x_tn1) / dt

            # for decode
            xhat_t = self.p_xhat.cond(x_tn1, u_tn1).rsample()
            self.p_decoder.cond(xhat_t)

            return 0, 0, 0, x_t, v_t

    @property
    def dim_xhat(self):
        return self._dim_xhat


# TODO:
class NewtonianVAEJIACell:
    pass


NewtonianVAECellSeries = Union[NewtonianVAECell, NewtonianVAEDerivationCell]


class Stepper:
    def __init__(self, cell: NewtonianVAECellSeries) -> None:
        self.cell = cell
        self._is_reset = False

    def reset(self):
        self._is_reset = True

    def step(self, u_tn1: Tensor, I_t: Tensor) -> Tuple[Tensor, Tensor]:
        assert u_tn1.ndim == 2 and I_t.ndim == 4
        assert u_tn1.shape[0] == I_t.shape[0]  # = N

        if self._is_reset:
            # shape (BS, dim(u))
            self.v_ = torch.zeros(
                size=(u_tn1.shape[1], u_tn1.shape[-1]), device=u_tn1.device, dtype=u_tn1.dtype
            )
            self.x_ = self.cell.q_encoder.cond(I_t).rsample()

        else:
            E, E_ll, E_kl, self.x_, self.v_ = self.cell(I_t, self.x_, u_tn1, self.v_, 0.1)

        return self.x_, self.v_


class CollectTimeSeriesData:
    def __init__(self, cell: NewtonianVAECellSeries, T: int, dtype: np.dtype) -> None:
        xp = np
        self.cell = cell

        self.LOG_x = xp.full((T, cell.dim_x), xp.nan, dtype=dtype)
        self.LOG_x_mean = xp.full((T, cell.dim_x), xp.nan, dtype=dtype)
        self.LOG_v = xp.full((T, cell.dim_x), xp.nan, dtype=dtype)
        self.LOG_I_dec = xp.full((T, 3, 64, 64), xp.nan, dtype=dtype)

        if type(cell) == NewtonianVAEDerivationCell:
            self.LOG_xhat_mean = xp.full((T, cell.dim_xhat), xp.nan, dtype=dtype)
            self.LOG_xhat_std = xp.full((T, cell.dim_xhat), xp.nan, dtype=dtype)

    def run(
        self,
        action: Tensor,
        observation: Tensor,
        device: torch.device,
        is_save=False,
    ):
        T = len(action)

        E_sum: Tensor = 0  # = Nagative ELBO
        LOG_E_ll_sum: Tensor = 0
        LOG_E_kl_sum: Tensor = 0

        for t in range(T):
            u_tn1, I_t = action[t].to(device), observation[t].to(device)

            if t == 0:
                # shape (BS, dim(u))
                v_t = torch.zeros(
                    size=(action.shape[1], action.shape[-1]), device=device, dtype=action.dtype
                )
                x_t = self.cell.q_encoder.cond(I_t).rsample()
            else:
                E, E_ll, E_kl, x_t, v_t = self.cell(I_t, x_tn1, u_tn1, v_tn1, 0.1)

                E_sum += E
                LOG_E_ll_sum += E_ll
                LOG_E_kl_sum += E_kl

                if is_save:
                    self.LOG_I_dec[t] = self.as_save(self.cell.p_decoder.decode())
                    if type(self.cell) == NewtonianVAEDerivationCell:
                        self.LOG_xhat_std[t] = self.as_save(self.cell.p_xhat.scale)
                        self.LOG_xhat_mean[t] = self.as_save(self.cell.p_xhat.loc)

            x_tn1, v_tn1 = x_t, v_t

            # DEBUG:
            # print(f"time: {t}")
            # tp_debug.check_dist_model(cell)

            if is_save:
                self.LOG_x[t] = self.as_save(x_t)
                self.LOG_v[t] = self.as_save(v_t)
                self.LOG_x_mean[t] = self.as_save(self.cell.q_encoder.loc)

        E_sum /= T
        LOG_E_ll_sum /= T
        LOG_E_kl_sum /= T

        return E_sum, LOG_E_ll_sum, LOG_E_kl_sum

    @staticmethod
    def as_save(x: Tensor):
        assert x.shape[0] == 1
        return x.detach().squeeze(0).cpu().numpy()
