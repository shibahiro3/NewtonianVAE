import dataclasses
from collections import OrderedDict
from numbers import Real
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import NumberType, Tensor, nn

import mypython.ai.torchprob as tp
import mypython.ai.torchprob.debug as tp_debug
from mypython.ai.util import find_function, swap01
from mypython.terminal import Color

from . import component


class NewtonianVAECellBase(nn.Module):
    """
    Classes that inherit from this class must not store variables from the previous time internally,
    as well as
    `RNNCell <https://pytorch.org/docs/stable/generated/torch.nn.RNNCell.html>`_,
    `LSTMCell <https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html>`_,
    and `GRUCell <https://pytorch.org/docs/stable/generated/torch.nn.GRUCell.html>`_.
    """

    def __init__(
        self,
        *,
        dim_x: int,
        regularization: bool = False,
    ):
        super().__init__()

        self.dim_x = dim_x
        self.regularization = regularization

        self.kl_beta = 1

    @property
    def decodable(self) -> bool:
        raise NotImplementedError()

    @staticmethod
    def img_reduction(x: Tensor):
        # x shape: (*, B, C, H, W)
        return x.sum(dim=(-3, -2, -1)).mean(dim=-1)

    @staticmethod
    def vec_reduction(x: Tensor):
        # x shape: (*, B, D)
        return x.sum(dim=-1).mean(dim=-1)


class NewtonianVAECell(NewtonianVAECellBase):
    def __init__(
        self,
        *,
        dim_x: int,
        velocity: dict,
        transition: dict,
        encoder: dict,
        decoder: dict,
        regularization: int = False,
    ) -> None:
        super().__init__(dim_x=dim_x, regularization=regularization)

        self.f_velocity = component.Velocity(dim_x=dim_x, **velocity)
        self.p_transition = component.Transition(**transition)  # prior
        self.q_encoder = component.Encoder(dim_x=dim_x, **encoder)  # posterior
        self.p_decoder = component.Decoder(dim_input=dim_x, **decoder)

    @dataclasses.dataclass
    class Pack:
        E: Tensor  # Use for training
        E_ll: Tensor
        E_kl: Tensor
        beta_kl: Tensor
        x_q_t: Tensor  # Use for training
        v_t: Tensor
        I_t_rec: Tensor

    def __call__(self, *args, **kwargs) -> Pack:
        return super().__call__(*args, **kwargs)

    def forward(self, I_t: Tensor, x_q_tn1: Tensor, x_q_tn2: Tensor, u_tn1: Tensor, dt: Tensor):
        """"""
        v_tn1 = (x_q_tn1 - x_q_tn2) / dt
        v_t = self.f_velocity(x_q_tn1, u_tn1, v_tn1, dt)
        x_p_t = self.p_transition.given(x_q_tn1, v_t, dt).rsample()

        # E_ll = self.img_reduction(tp.log(self.p_decoder, I_t).given(x_p_t))
        ### log p(It | x_p_t) (mu)
        I_t_rec = self.p_decoder(x_p_t)
        # log Normal Dist.: -0.5 * (((x - mu) / sigma) ** 2 ...
        E_ll = -self.img_reduction(F.mse_loss(I_t_rec, I_t, reduction="none"))
        ###

        E_kl = self.vec_reduction(tp.KLdiv(self.q_encoder.given(I_t), self.p_transition))
        beta_kl = self.kl_beta * E_kl
        E = E_ll - beta_kl

        if self.regularization:
            E -= self.vec_reduction(tp.KLdiv(self.q_encoder, tp.Normal01))

        x_q_t = self.q_encoder.rsample()

        return self.Pack(
            E=E,
            x_q_t=x_q_t,
            E_ll=E_ll.detach(),
            E_kl=E_kl.detach(),
            beta_kl=beta_kl.detach(),
            v_t=v_t.detach(),
            I_t_rec=I_t_rec.detach(),
        )


class MNVAECell(NewtonianVAECellBase):  # TODO: NewtonianVAECellBase (dim_x=...)
    """
    Based on NewtonianVAECell
    """

    def __init__(
        self,
        dim_x: int,
        velocity: dict,
        transition: dict,
        encoder: dict,
        decoder: dict,
        regularization: bool = False,
        decoder_free: bool = False,
    ) -> None:
        super().__init__(dim_x=dim_x, regularization=regularization)

        self._decoder_free = decoder_free
        # self.force_training = False  # for add_graph of tensorboard
        self.alpha = 1

        self.f_velocity = component.Velocity(dim_x=dim_x, **velocity)
        self.p_transition = component.Transition(**transition)
        self.q_encoder = component.MultiEncoder(dim_x=dim_x, **encoder)
        self.p_decoder = component.MultiDecoder(dim_x=dim_x, **decoder)

    @staticmethod
    def img_reduction(x: Tensor):
        # x shape: (*, B, C, H, W)
        return x.sum(dim=(-3, -2, -1)).mean(dim=-1)

    @staticmethod
    def vec_reduction(x: Tensor):
        # x shape: (*, B, D)
        return x.sum(dim=-1).mean(dim=-1)

    def set_decoder_free(self, mode: bool):
        self._decoder_free = mode
        if mode:
            self.p_decoder.requires_grad_(not mode)  # just in case

    def __call__(self, *args, **kwargs) -> Dict[str, Tensor]:
        return super().__call__(*args, **kwargs)

    def forward(
        self, I_t: List[Tensor], x_q_tn1: Tensor, x_q_tn2: Tensor, u_tn1: Tensor, dt: Tensor
    ):
        """"""

        if self.alpha == 0:
            self.set_decoder_free(True)

        v_tn1 = (x_q_tn1 - x_q_tn2) / dt
        v_t = self.f_velocity(x_q_tn1, u_tn1, v_tn1, dt)
        x_p_t = self.p_transition.given(x_q_tn1, v_t, dt).rsample()

        recs = 0
        if not self._decoder_free:
            ### log p(It | x_p_t)
            I_t_recs = self.p_decoder(x_p_t)  # only here
            image_losses = []
            for i in range(len(I_t)):
                # log Normal Dist.: -0.5 * (((x - mu) / sigma) ** 2 ...
                loss = self.img_reduction(F.mse_loss(I_t_recs[i], I_t[i], reduction="none"))
                recs -= loss
                image_losses.append(loss.item())
            ###

        E_kl = self.vec_reduction(tp.KLdiv(self.q_encoder.given(I_t), self.p_transition))
        beta_kl = self.kl_beta * E_kl
        E = self.alpha * recs - beta_kl

        if self.regularization:
            E -= self.vec_reduction(tp.KLdiv(self.q_encoder, tp.Normal01))

        x_q_t = self.q_encoder.rsample()

        ret = dict(
            E=E,
            x_q_t=x_q_t,
            v_t=v_t.detach(),
            KL_loss=E_kl.item(),
            beta_kl=beta_kl.detach(),
        )

        if not self._decoder_free:
            ret["image_losses"] = image_losses
            ret["I_t_recs"] = I_t_recs

        return ret
