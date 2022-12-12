"""
x_t    == x_t
x_tn1  == x_{t-1}
x_tn2  == x_{t-2}
x_tp1  == x_{t+1}
x_tp2  == x_{t+2}
"""

from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import NumberType, Tensor, nn

import mypython.ai.torchprob as tp
import mypython.ai.torchprob.debug as tp_debug
from mypython.ai.torch_util import find_function
from mypython.terminal import Color

from .component import Decoder, Encoder, Pxhat, PXmiddleCat, Transition, Velocity


class NewtonianVAECellBase(nn.Module):
    """
    Classes that inherit from this class should not store variables from the previous time internally,
    as well as
    `RNNCell <https://pytorch.org/docs/stable/generated/torch.nn.RNNCell.html>`_,
    `LSTMCell <https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html>`_,
    and `GRUCell <https://pytorch.org/docs/stable/generated/torch.nn.GRUCell.html>`_.
    """

    def __init__(
        self,
        dim_x: int,
        transition_std: float,
        encoder_dim_middle: int,
        encoder_std_function: str,
        fix_abc: Union[None, Tuple[NumberType, NumberType, NumberType]] = None,
        regularization: bool = False,
    ):
        super().__init__()

        self.regularization = regularization
        self.kl_beta = 1
        self.force_training = False  # for add_graph of tensorboard

        # v_t = v_{t-1} + Δt・(Ax_{t-1} + Bv_{t-1} + Cu_{t-1})
        self.f_velocity = Velocity(dim_x, fix_abc)

        # p(x_t | x_{t-1}, u_{t-1}; v_t)
        self.p_transition = Transition(transition_std)

        # q(x_t | I_t) (posterior)
        encoder_std_function = find_function(encoder_std_function)
        self.q_encoder = Encoder(
            dim_x, dim_middle=encoder_dim_middle, std_function=encoder_std_function
        )

        self._dim_x = dim_x

    @property
    def dim_x(self):
        return self._dim_x

    @property
    def info(self) -> set:
        return set()

    class Pack:
        def __init__(self, E: Tensor, E_ll: Tensor, E_kl: Tensor, x_t: Tensor, v_t: Tensor):
            self.E = E  # Use for training
            self.E_ll = E_ll
            self.E_kl = E_kl
            self.x_t = x_t  # Use for training
            self.v_t = v_t  # Use for training

    def __call__(self, *args, **kwargs) -> Pack:
        return super().__call__(*args, **kwargs)


class NewtonianVAECell(NewtonianVAECellBase):
    """
    Eq (11)
    """

    def __init__(self, dim_x, *args, **kwargs) -> None:
        super().__init__(dim_x=dim_x, *args, **kwargs)

        # p(I_t | x_t)
        self.p_decoder = Decoder(dim_x=dim_x)

    def forward(self, I_t: Tensor, x_tn1: Tensor, u_tn1: Tensor, v_tn1: Tensor, dt: Tensor):
        """"""

        if self.training or self.force_training:
            v_t = self.f_velocity(x_tn1, u_tn1, v_tn1, dt)
            x_t = self.p_transition.cond(x_tn1, v_t, dt).rsample()
            # *_t について、各次元(ピクセル)についてはsum, batch size については mean
            E_ll = tp.log(self.p_decoder, I_t, x_t).sum(dim=(1, 2, 3)).mean(dim=0)
            E_kl = tp.KLdiv(self.q_encoder.cond(I_t), self.p_transition).sum(dim=1).mean(dim=0)
            E = E_ll - self.kl_beta * E_kl

            if self.regularization:
                E -= tp.KLdiv(self.q_encoder, tp.Normal01).sum(dim=1).mean(dim=0)

            return super().Pack(E=E, E_ll=E_ll.detach(), E_kl=E_kl.detach(), x_t=x_t, v_t=v_t)

        else:
            x_t = self.q_encoder.cond(I_t).rsample()
            self.p_decoder.cond(x_t)  # for cell.p_decoder.decode()
            v_t = (x_t - x_tn1) / dt  # for only visualize
            return super().Pack(E=0, E_ll=0, E_kl=0, x_t=x_t, v_t=v_t)

    @property
    def info(self) -> str:
        return {"V1"}


class NewtonianVAEDerivationCell(NewtonianVAECellBase):
    """
    Eq (23)
    """

    def __init__(
        self,
        dim_x: int,
        dim_xhat: int,
        dim_pxhat_middle: int,
        pxhat_std_function: str,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(dim_x=dim_x, *args, **kwargs)

        self._dim_xhat = dim_xhat

        # p(I_t | xhat_t)
        self.p_decoder = Decoder(dim_x=dim_xhat)

        # p(xhat_t | x_{t-1}, u_{t-1})
        pxhat_std_function = find_function(pxhat_std_function)
        self.p_xhat = Pxhat(dim_x, dim_xhat, dim_pxhat_middle, std_function=pxhat_std_function)

    def forward(self, I_t: Tensor, x_tn1: Tensor, u_tn1: Tensor, v_tn1: Tensor, dt: Tensor):
        """"""

        if self.training or self.force_training:
            v_t = self.f_velocity(x_tn1, u_tn1, v_tn1, dt)
            xhat_t = self.p_xhat.cond(x_tn1, u_tn1).rsample()
            E_ll = tp.log(self.p_decoder, I_t, xhat_t).sum(dim=(1, 2, 3)).mean(dim=0)
            E_kl = (
                tp.KLdiv(self.q_encoder.cond(I_t), self.p_transition.cond(x_tn1, v_t, dt))
                .sum(dim=1)
                .mean(dim=0)
            )
            E = E_ll - self.kl_beta * E_kl

            if self.regularization:
                E -= tp.KLdiv(self.q_encoder, tp.Normal01).sum(dim=1).mean(dim=0)

            x_t = self.q_encoder.rsample()

            return super().Pack(E=E, E_ll=E_ll.detach(), E_kl=E_kl.detach(), x_t=x_t, v_t=v_t)

        else:
            x_t = self.q_encoder.cond(I_t).rsample()
            xhat_t = self.p_xhat.cond(x_tn1, u_tn1).rsample()
            v_t = (x_t - x_tn1) / dt

            self.p_decoder.cond(xhat_t)  # for cell.p_decoder.decode()
            return super().Pack(E=0, E_ll=0, E_kl=0, x_t=x_t, v_t=v_t)

    @property
    def dim_xhat(self):
        return self._dim_xhat

    @property
    def info(self) -> str:
        return {"V1", "xhat"}


class NewtonianVAEBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.is_save = False

        self.cell: Union[NewtonianVAECell, NewtonianVAEDerivationCell]

    def forward(self, action: Tensor, observation: Tensor, delta: Tensor):
        """"""

        T = len(action)

        self._init_LOG(T=T, dtype=action.dtype)

        E_sum: Tensor = 0  # = Nagative ELBO
        E_ll_sum: Tensor = 0  # Not use for training
        E_kl_sum: Tensor = 0  # Not use for training

        for t in range(T):
            u_tn1, I_t = action[t], observation[t]

            if t == 0:
                BS, D = action.shape[1], action.shape[-1]
                v_t = torch.zeros(size=(BS, D), device=action.device, dtype=action.dtype)
                x_t = self.cell.q_encoder.cond(I_t).rsample()
                x_tn1 = x_t
                v_tn1 = v_t

            else:
                output = self.cell(I_t=I_t, x_tn1=x_tn1, u_tn1=u_tn1, v_tn1=v_tn1, dt=delta[t])

                E_sum += output.E
                E_ll_sum += output.E_ll
                E_kl_sum += output.E_kl

                if self.is_save:
                    self.LOG_I_dec[t] = as_save(self.cell.p_decoder.decode())
                    if "xhat" in self.cell.info:
                        self.LOG_xhat_std[t] = as_save(self.cell.p_xhat.scale)
                        self.LOG_xhat_mean[t] = as_save(self.cell.p_xhat.loc)

                x_tn1 = output.x_t
                v_tn1 = output.v_t

            # ### DEBUG:
            # print(f"time: {t}")
            # tp_debug.check_dist_model(self.cell)

            if self.is_save:
                self.LOG_x[t] = as_save(x_t)
                self.LOG_v[t] = as_save(v_t)
                self.LOG_x_mean[t] = as_save(self.cell.q_encoder.loc)

        E = E_sum / T
        E_ll = E_ll_sum / T
        E_kl = E_kl_sum / T

        return E, E_ll, E_kl

    def __call__(self, *args, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        return super().__call__(*args, **kwargs)

    def _init_LOG(self, T: int, dtype: torch.dtype):
        xp = np
        dtype = torch.empty((), dtype=dtype).numpy().dtype

        self.LOG_x = xp.full((T, self.cell.dim_x), xp.nan, dtype=dtype)
        self.LOG_x_mean = xp.full((T, self.cell.dim_x), xp.nan, dtype=dtype)
        self.LOG_v = xp.full((T, self.cell.dim_x), xp.nan, dtype=dtype)
        self.LOG_I_dec = xp.full((T, 3, 64, 64), xp.nan, dtype=dtype)

        if "xhat" in self.cell.info:
            self.LOG_xhat_mean = xp.full((T, self.cell.dim_xhat), xp.nan, dtype=dtype)
            self.LOG_xhat_std = xp.full((T, self.cell.dim_xhat), xp.nan, dtype=dtype)


class NewtonianVAE(NewtonianVAEBase):
    r"""Computes ELBO based on formula (11).

    Computes according to the following formula:

    .. math::
        \begin{array}{ll} \\
            A = f(\x_{t-1}, \v_{t-1}, \u_{t-1}) \\
            B = -\log f(\x_{t-1}, \v_{t-1}, \u_{t-1}) \\
            C = \log f(\x_{t-1}, \v_{t-1}, \u_{t-1}) \\
            \v_t = \v_{t-1} + \Delta t \cdot (A\x_{t-1} + B\v_{t-1} + C\u_{t-1}) \\
            \x_{t} \sim p(\x_t \mid \x_{t-1}, \u_{t-1}; \v_{t}) \\
            E = \displaystyle \sum_t \left( \log (\I_t \mid \x_t) - \KL{q(\x_t \mid \I_t)}{p(\x_t \mid \x_{t-1}, \u_{t-1}; \v_{t})} \right)
        \end{array}

    where:

    .. math::
        \begin{array}{ll}
            \v_0 = \boldsymbol{0} \\
            \x_0 \sim q(\x_0 \mid \I_0) \\
            p(\x_t \mid \x_{t-1}, \u_{t-1}) = \mathcal{N}(\x_t \mid \x_{t-1} + \Delta t \cdot \v_t, \sigma^2) \\
            \x_{t-2} \leftarrow \x_{t-1}
        \end{array}

    During evaluation:

    .. math::
        \begin{array}{ll}
            \v_{t-1} = (\x_{t-1} - \x_{t-2}) / \Delta t
        \end{array}


    Inputs: action, observation, dt
        * **action**: tensor of shape :math:`(T, N, D)`
        * **observation**: tensor of shape :math:`(T, N, 3, 64, 64)`
        * **dt**: tensor of shape :math:`(T)`

        where:

        .. math::
            \begin{aligned}
                N ={} & \text{batch size} \\
                T ={} & \text{sequence length} \\
                D ={} & \mathrm{dim}(\u) \\
            \end{aligned}

    References in paper:
        * During inference, vt is computed as 
          vt = (xt − xt−1)/∆t, with xt ∼ q(xt|It) and xt−1 ∼ q(xt−1|It−1).
        * we added an additional regularization term to the latent space, KL(q(x|I)‖N (0, 1))
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        self.cell = NewtonianVAECell(*args, **kwargs)


class NewtonianVAEDerivation(NewtonianVAEBase):
    r"""Computes ELBO based on formula (23).

    Computes according to the following formula:

    .. math::
        \begin{array}{ll} \\
            \v_t = \v_{t-1} + \Delta t \cdot (A\x_{t-1} + B\v_{t-1} + C\u_{t-1}) \\
            \x_{t} \sim p(\x_t \mid \x_{t-1}, \u_{t-1}; \v_{t}) \\
            \xhat_{t} \sim p(\xhat_t \mid \x_{t-1}, \u_{t-1}) \\
            E = \displaystyle \sum_t \left( \log (\I_t \mid \xhat_t) - \KL{q(\x_t \mid \I_t)}{p(\x_t \mid \x_{t-1}, \u_{t-1}; \v_{t})} \right)
        \end{array}

    where:

    .. math::
        \begin{array}{ll}
            \v_0 = \boldsymbol{0} \\
            \x_0 \sim q(\x_0 \mid \I_0) \\
            p(\x_t \mid \x_{t-1}, \u_{t-1}) = \mathcal{N}(\x_t \mid \x_{t-1} + \Delta t \cdot \v_t, \sigma^2) \\
            \v_{t} \leftarrow \v_{t-1} \\
            \x_{t} \leftarrow \x_{t-1}
        \end{array}

    During evaluation:

    .. math::
        \begin{array}{ll}
            \v_{t-1} = (\x_{t-1} - \x_{t-2}) / \Delta t
        \end{array}


    Inputs: action, observation, dt
        * **action**: tensor of shape :math:`(T, N, D)`
        * **observation**: tensor of shape :math:`(T, N, 3, 64, 64)`
        * **dt**: tensor of shape :math:`(T)`

        where:

        .. math::
            \begin{aligned}
                N ={} & \text{batch size} \\
                T ={} & \text{sequence length} \\
                D ={} & \mathrm{dim}(\u) \\
            \end{aligned}

    References in paper:
        * During inference, vt is computed as 
          vt = (xt − xt−1)/∆t, with xt ∼ q(xt|It) and xt−1 ∼ q(xt−1|It−1).
        * we added an additional regularization term to the latent space, KL(q(x|I)‖N (0, 1))
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        self.cell = NewtonianVAEDerivationCell(*args, **kwargs)


######################################## V2 ########################################


class NewtonianVAEV2CellBase(NewtonianVAECellBase):
    def __init__(self, dim_x, *args, **kwargs) -> None:
        super().__init__(dim_x=dim_x, *args, **kwargs)

    class Pack:
        def __init__(self, E: Tensor, E_ll: Tensor, E_kl: Tensor, x_q_t: Tensor, v_t: Tensor):
            self.E = E  # Use for training
            self.E_ll = E_ll
            self.E_kl = E_kl
            self.x_q_t = x_q_t  # Use for training
            self.v_t = v_t

    def __call__(self, *args, **kwargs) -> Pack:
        return super().__call__(*args, **kwargs)


class NewtonianVAEV2Cell(NewtonianVAEV2CellBase):
    def __init__(self, dim_x, *args, **kwargs) -> None:
        super().__init__(dim_x=dim_x, *args, **kwargs)

        # p(I_t | x_t)
        self.p_decoder = Decoder(dim_x=dim_x)

    def forward(self, I_t: Tensor, x_q_tn1: Tensor, x_q_tn2: Tensor, u_tn1: Tensor, dt: Tensor):
        """"""
        v_tn1 = (x_q_tn1 - x_q_tn2) / dt
        v_t = self.f_velocity(x_q_tn1, u_tn1, v_tn1, dt)
        x_p_t = self.p_transition.cond(x_q_tn1, v_t, dt).rsample()
        E_ll = self.img_reduction(tp.log(self.p_decoder, I_t, x_p_t))
        E_kl = self.vec_reduction(tp.KLdiv(self.q_encoder.cond(I_t), self.p_transition))
        E = E_ll - self.kl_beta * E_kl

        if self.regularization:
            E -= self.vec_reduction(tp.KLdiv(self.q_encoder, tp.Normal01))

        x_q_t = self.q_encoder.rsample()

        return super().Pack(
            E=E,
            x_q_t=x_q_t,
            E_ll=E_ll.detach(),
            E_kl=E_kl.detach(),
            v_t=v_t.detach(),
        )

    @staticmethod
    def img_reduction(x: Tensor):
        return x.sum(dim=(1, 2, 3)).mean(dim=0)

    @staticmethod
    def vec_reduction(x: Tensor):
        return x.sum(dim=1).mean(dim=0)

    @property
    def info(self) -> str:
        return {"V2"}


class NewtonianVAEV2DerivationCell(NewtonianVAEV2CellBase):
    """
    Deprecated

    Eq (23)
    """

    def __init__(
        self,
        dim_x: int,
        dim_xhat: int,
        dim_pxhat_middle: int,
        pxhat_std_function: str,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(dim_x=dim_x, *args, **kwargs)

        self._dim_xhat = dim_xhat

        # p(I_t | xhat_t)
        self.p_decoder = Decoder(dim_x=dim_xhat)

        # p(xhat_t | x_{t-1}, u_{t-1})
        pxhat_std_function = find_function(pxhat_std_function)
        self.p_xhat = Pxhat(dim_x, dim_xhat, dim_pxhat_middle, std_function=pxhat_std_function)

    def forward(self, I_tn1: Tensor, I_t: Tensor, x_tn2: Tensor, u_tn1: Tensor, dt: Tensor):
        x_tn1 = self.q_encoder.cond(I_tn1).rsample()
        v_tn1 = (x_tn1 - x_tn2) / dt
        v_t: Tensor = self.f_velocity(x_tn1, u_tn1, v_tn1, dt)
        xhat_t = self.p_xhat.cond(x_tn1, u_tn1).rsample()
        E_ll = tp.log(self.p_decoder, I_t, xhat_t).sum(dim=(1, 2, 3)).mean(dim=0)
        E_kl = (
            tp.KLdiv(self.q_encoder.cond(I_t), self.p_transition.cond(x_tn1, v_t, dt))
            .sum(dim=1)
            .mean(dim=0)
        )
        E = E_ll - self.kl_beta * E_kl

        if self.regularization:
            E -= tp.KLdiv(self.q_encoder, tp.Normal01).sum(dim=1).mean(dim=0)

        x_t = self.q_encoder.rsample()

        return super().Pack(
            E=E,
            E_ll=E_ll.detach(),
            E_kl=E_kl.detach(),
            x_tn1=x_tn1,
            x_q_t=x_t,
            v_t=v_t.detach(),
        )

    @property
    def dim_xhat(self):
        return self._dim_xhat

    @property
    def info(self) -> str:
        return {"V2", "xhat"}


class NewtonianVAEV2Base(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.is_save = False

        self.cell: Union[NewtonianVAEV2Cell, NewtonianVAEV2DerivationCell]

    def forward(self, action: Tensor, observation: Tensor, delta: Tensor):
        """"""

        T = len(action)

        self._init_LOG(T=T, dtype=action.dtype)

        E_sum: Tensor = 0  # = Nagative ELBO
        E_ll_sum: Tensor = 0
        E_kl_sum: Tensor = 0

        for t in range(T):
            u_tn1, I_t = action[t], observation[t]

            if t == 0:
                BS, D = action.shape[1], action.shape[-1]

                x_q_t = self.cell.q_encoder.cond(I_t).rsample()
                v_t = torch.zeros(size=(BS, D), device=action.device, dtype=action.dtype)
                I_dec = self.cell.p_decoder.cond(x_q_t).decode()

                x_q_tn1 = x_q_t

            elif t == 1:
                BS, D = action.shape[1], action.shape[-1]

                x_q_t = self.cell.q_encoder.cond(I_t).rsample()
                v_t = (x_q_t - x_q_tn1) / delta[t]
                I_dec = self.cell.p_decoder.cond(x_q_t).decode()

            else:
                output = self.cell(
                    I_t=I_t, x_q_tn1=x_q_tn1, x_q_tn2=x_q_tn2, u_tn1=u_tn1, dt=delta[t]
                )

                E_sum += output.E
                E_ll_sum += output.E_ll
                E_kl_sum += output.E_kl

                x_q_t = output.x_q_t
                v_t = output.v_t
                I_dec = self.cell.p_decoder.decode()

                if self.is_save:
                    if "xhat" in self.cell.info:
                        self.LOG_xhat_std[t] = as_save(self.cell.p_xhat.scale)
                        self.LOG_xhat_mean[t] = as_save(self.cell.p_xhat.loc)

            x_q_tn2 = x_q_tn1
            x_q_tn1 = x_q_t

            # ###### DEBUG:
            # print(f"time: {t}")
            # tp_debug.check_dist_model(self.cell)

            if self.is_save:
                self.LOG_x[t] = as_save(x_q_t)
                self.LOG_v[t] = as_save(v_t)
                self.LOG_I_dec[t] = as_save(I_dec)
                self.LOG_x_mean[t] = as_save(self.cell.q_encoder.loc)

        E = E_sum / T
        E_ll = E_ll_sum / T
        E_kl = E_kl_sum / T

        return E, E_ll, E_kl

    def __call__(self, *args, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        return super().__call__(*args, **kwargs)

    def _init_LOG(self, T: int, dtype: torch.dtype):
        xp = np
        dtype = torch.empty((), dtype=dtype).numpy().dtype

        self.LOG_x = xp.full((T, self.cell.dim_x), xp.nan, dtype=dtype)
        self.LOG_x_mean = xp.full((T, self.cell.dim_x), xp.nan, dtype=dtype)
        self.LOG_v = xp.full((T, self.cell.dim_x), xp.nan, dtype=dtype)
        self.LOG_I_dec = xp.full((T, 3, 64, 64), xp.nan, dtype=dtype)

        if "xhat" in self.cell.info:
            self.LOG_xhat_mean = xp.full((T, self.cell.dim_xhat), xp.nan, dtype=dtype)
            self.LOG_xhat_std = xp.full((T, self.cell.dim_xhat), xp.nan, dtype=dtype)


class NewtonianVAEV2(NewtonianVAEV2Base):
    r"""Computes ELBO based on formula (11).

    This implementation was based on Mr. `Ito <https://github.com/ItoMasaki>`_'s opinion.

    Computes according to the following formula:

    .. math::
        \begin{array}{ll} \\
            \x_{t-1} \sim q(\x_{t-1} \mid \I_{t-1}) \\
            \v_{t-1} = (\x_{t-1} - \x_{t-2}) / \Delta t \\
            A = \mathrm{diag}(f(\x_{t-1}, \v_{t-1}, \u_{t-1})) \\
            B = -\log \mathrm{diag}(f(\x_{t-1}, \v_{t-1}, \u_{t-1})) \\
            C = \log \mathrm{diag}(f(\x_{t-1}, \v_{t-1}, \u_{t-1})) \\
            \v_t = \v_{t-1} + \Delta t \cdot (A\x_{t-1} + B\v_{t-1} + C\u_{t-1}) \\
            \x_{t} \sim p(\x_t \mid \x_{t-1}, \u_{t-1}; \v_{t}) \\
            ELBO = \displaystyle \sum_t \left( \log (\I_t \mid \x_t) - \KL{q(\x_t \mid \I_t)}{p(\x_t \mid \x_{t-1}, \u_{t-1}; \v_{t})} \right)
        \end{array}

    where:

    .. math::
        \begin{array}{ll}
            p(\x_t \mid \x_{t-1}, \u_{t-1}) = \mathcal{N}(\x_t \mid \x_{t-1} + \Delta t \cdot \v_t, \sigma^2) \\
            \x_{t-2} \leftarrow \x_{t-1}
        \end{array}

    The initial values follow the formula below:

    .. math::
        \begin{array}{ll}
            \v_0 = \boldsymbol{0} \\
            \x_0 \sim q(\x_0 \mid \I_0)
        \end{array}

    LOG_x is collected about :math:`\x_{t} \sim q(\x_t \mid \I_t)`.
    
    Inputs: action, observation, dt
        * **action**: tensor of shape :math:`(T, N, D)`
        * **observation**: tensor of shape :math:`(T, N, 3, 64, 64)`
        * **dt**: tensor of shape :math:`(T)`

        where:

        .. math::
            \begin{aligned}
                N ={} & \text{batch size} \\
                T ={} & \text{sequence length} \\
                D ={} & \mathrm{dim}(\u) \\
            \end{aligned}

    References in paper:
        * During inference, vt is computed as 
          vt = (xt − xt−1)/∆t, with xt ∼ q(xt|It) and xt−1 ∼ q(xt−1|It−1).
        * we added an additional regularization term to the latent space, KL(q(x|I)‖N (0, 1))
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        self.cell = NewtonianVAEV2Cell(*args, **kwargs)


class NewtonianVAEV2Derivation(NewtonianVAEV2Base):
    r"""Computes ELBO based on formula (23).

    Computes according to the following formula:

    .. math::
        \begin{array}{ll} \\
            \x_{t-1} \sim q(\x_{t-1} \mid \I_{t-1}) \\
            \v_{t-1} = (\x_{t-1} - \x_{t-2}) / \Delta t \\
            A = \mathrm{diag}(f(\x_{t-1}, \v_{t-1}, \u_{t-1})) \\
            B = -\log \mathrm{diag}(f(\x_{t-1}, \v_{t-1}, \u_{t-1})) \\
            C = \log \mathrm{diag}(f(\x_{t-1}, \v_{t-1}, \u_{t-1})) \\
            \v_t = \v_{t-1} + \Delta t \cdot (A\x_{t-1} + B\v_{t-1} + C\u_{t-1}) \\
            \xhat_{t} \sim p(\xhat_t \mid \x_{t-1}, \u_{t-1}) \\
            ELBO = \displaystyle \sum_t \left( \log (\I_t \mid \xhat_t) - \KL{q(\x_t \mid \I_t)}{p(\x_t \mid \x_{t-1}, \u_{t-1}; \v_{t})} \right)
        \end{array}

    where:

    .. math::
        \begin{array}{ll}
            p(\x_t \mid \x_{t-1}, \u_{t-1}) = \mathcal{N}(\x_t \mid \x_{t-1} + \Delta t \cdot \v_t, \sigma^2) \\
            \x_{t-2} \leftarrow \x_{t-1}
        \end{array}

    The initial values follow the formula below:

    .. math::
        \begin{array}{ll}
            \v_0 = \boldsymbol{0} \\
            \x_0 \sim q(\x_0 \mid \I_0)
        \end{array}

    LOG_x is collected about :math:`\x_{t} \sim q(\x_t \mid \I_t)`.

    Inputs: action, observation, dt
        * **action**: tensor of shape :math:`(T, N, D)`
        * **observation**: tensor of shape :math:`(T, N, 3, 64, 64)`
        * **dt**: tensor of shape :math:`(T)`

        where:

        .. math::
            \begin{aligned}
                N ={} & \text{batch size} \\
                T ={} & \text{sequence length} \\
                D ={} & \mathrm{dim}(\u) \\
            \end{aligned}

    References in paper:
        * During inference, vt is computed as 
          vt = (xt − xt−1)/∆t, with xt ∼ q(xt|It) and xt−1 ∼ q(xt−1|It−1).
        * we added an additional regularization term to the latent space, KL(q(x|I)‖N (0, 1))
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        self.cell = NewtonianVAEV2DerivationCell(*args, **kwargs)


_NewtonianVAECellFamily = [
    NewtonianVAECell,
    NewtonianVAEDerivationCell,
    NewtonianVAEV2Cell,
    NewtonianVAEV2DerivationCell,
]
NewtonianVAECellFamily = Union[
    NewtonianVAECell,
    NewtonianVAEDerivationCell,
    NewtonianVAEV2Cell,
    NewtonianVAEV2DerivationCell,
]
_NewtonianVAEFamily = [
    NewtonianVAE,
    NewtonianVAEDerivation,
    NewtonianVAEV2,
    NewtonianVAEV2Derivation,
]
NewtonianVAEFamily = Union[
    NewtonianVAE,
    NewtonianVAEDerivation,
    NewtonianVAEV2,
    NewtonianVAEV2Derivation,
]


class Stepper:
    class Pack:
        def __init__(self, I_t_dec: Tensor, x_t: Tensor, v_t: Tensor) -> None:
            self.I_t_dec = I_t_dec
            self.x_t = x_t
            self.v_t = v_t

    def __init__(self, cell: NewtonianVAECellFamily) -> None:
        self.cell = cell
        self._is_reset = False

        if "V1" in cell.info:
            self.step = self._stepV1
        elif "V2" in cell.info:
            self.step = self._stepV2
        else:
            assert False

    def reset(self):
        self._is_reset = True

    def _stepV1(self, u_tn1: Tensor, I_t: Tensor, dt: Tensor):
        assert u_tn1.ndim >= 2 and I_t.ndim >= 4
        assert u_tn1.shape[0] == I_t.shape[0]  # = N

        # NewtonianVAEBase # ref

        if self._is_reset:
            BS, D = u_tn1.shape[1], u_tn1.shape[-1]
            v_t = torch.zeros(size=(BS, D), device=u_tn1.device, dtype=u_tn1.dtype)
            x_t = self.cell.q_encoder.cond(I_t).rsample()
            self.x_ = x_t
            self.v_ = v_t

        else:
            output = self.cell(I_t=I_t, x_tn1=self.x_, u_tn1=u_tn1, v_tn1=self.v_, dt=dt)
            self.x_ = output.x_t
            self.v_ = output.v_t

        I_t_dec = self.cell.p_decoder.decode()
        return self.Pack(I_t_dec=I_t_dec, x_t=self.x_, v_t=self.v_)

    def _stepV2(self, u_tn1: Tensor, I_t: Tensor, dt: Tensor):
        assert u_tn1.ndim >= 2 and I_t.ndim >= 4
        assert u_tn1.shape[0] == I_t.shape[0]  # = N

        # NewtonianVAEV2Base # ref

        if self._is_reset:
            BS, D = u_tn1.shape[1], I_t.shape[-1]
            v_t = torch.zeros(size=(BS, D), device=u_tn1.device, dtype=u_tn1.dtype)
            x_t = self.cell.q_encoder.cond(I_t).rsample()

            self.x_tn2 = x_t

        else:
            output = self.cell(I_tn1=self.I_tn1, I_t=I_t, x_tn2=self.x_tn2, u_tn1=u_tn1, dt=dt)

            self.x_tn2 = output.x_tn1

        self.I_tn1 = I_t
        I_t_dec = self.cell.p_decoder.decode()
        return self.Pack(I_t_dec=I_t_dec, x_t=output.x_t, v_t=output.v_t)


def as_save(x: Tensor):
    assert x.shape[0] == 1  # N (BS)
    return x.detach().squeeze(0).cpu().numpy()


def get_NewtonianVAECell(name: str, *args, **kwargs) -> NewtonianVAECellFamily:
    for T in _NewtonianVAECellFamily:
        if name == T.__name__ or name == T.__name__[:-4]:  # 4 : "Cell"
            return T(*args, **kwargs)
    assert False


def get_NewtonianVAE(name: str, *args, **kwargs) -> NewtonianVAEFamily:
    for T in _NewtonianVAEFamily:
        if name == T.__name__ or name == T.__name__ + "Cell":
            return T(*args, **kwargs)
    assert False
