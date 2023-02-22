"""
x_t    == x_t
x_tn1  == x_{t-1}
x_tn2  == x_{t-2}
x_tp1  == x_{t+1}
x_tp2  == x_{t+2}
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import NumberType, Tensor, nn

import mypython.ai.torchprob as tp
import mypython.ai.torchprob.debug as tp_debug
from mypython.ai.util import find_function, swap01, to_np
from mypython.terminal import Color

from .cell import (
    MNVAECell,
    NewtonianVAECell,
    NewtonianVAEDerivationCell,
    NewtonianVAEV2Cell,
    NewtonianVAEV2DerivationCell,
)


class NewtonianVAEBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.is_save = True

    def __call__(self, *args, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        return super().__call__(*args, **kwargs)

    def init_LOG(self):
        # (T, B, D)

        self.LOG_x = []
        self.LOG_x_mean = []
        self.LOG_v = []
        self.LOG_I_dec = []

    def LOG2numpy(self, batch_first=False, squeezeN1=False):
        """
        if batch_first is True:
            (N, T, *)
        else:
            (T, N, *)
        """

        self.LOG_x = np.array(self.LOG_x)
        self.LOG_x_mean = np.array(self.LOG_x_mean)
        self.LOG_v = np.array(self.LOG_v)
        self.LOG_I_dec = np.array(self.LOG_I_dec)

        if squeezeN1:
            self.LOG_x = self.LOG_x.squeeze(1)
            self.LOG_x_mean = self.LOG_x_mean.squeeze(1)
            self.LOG_v = self.LOG_v.squeeze(1)
            self.LOG_I_dec = self.LOG_I_dec.squeeze(1)

        elif batch_first:
            self.LOG_x = swap01(self.LOG_x)
            self.LOG_x_mean = swap01(self.LOG_x_mean)
            self.LOG_v = swap01(self.LOG_v)
            self.LOG_I_dec = swap01(self.LOG_I_dec)

        # Color.print(self.LOG_x.shape)
        # Color.print(self.LOG_x_mean.shape)
        # Color.print(self.LOG_v.shape)
        # Color.print(self.LOG_I_dec.shape)


class NewtonianVAEDerivationBase(NewtonianVAEBase):
    def __init__(self) -> None:
        super().__init__()

    def init_LOG(self):
        # (T, B, D)

        self.LOG_x = []
        self.LOG_x_mean = []
        self.LOG_v = []
        self.LOG_I_dec = []
        self.LOG_xhat_mean = []
        self.LOG_xhat_std = []

    def LOG2numpy(self, batch_first=False, squeezeN1=False):
        """
        if batch_first is True:
            (N, T, *)
        else:
            (T, N, *)
        """

        self.LOG_x = np.array(self.LOG_x)
        self.LOG_x_mean = np.array(self.LOG_x_mean)
        self.LOG_v = np.array(self.LOG_v)
        self.LOG_I_dec = np.array(self.LOG_I_dec)
        self.LOG_xhat_mean = np.array(self.LOG_xhat_mean)
        self.LOG_xhat_std = np.array(self.LOG_xhat_std)

        if squeezeN1:
            self.LOG_x = self.LOG_x.squeeze(1)
            self.LOG_x_mean = self.LOG_x_mean.squeeze(1)
            self.LOG_v = self.LOG_v.squeeze(1)
            self.LOG_I_dec = self.LOG_I_dec.squeeze(1)
            self.LOG_xhat_mean = self.LOG_xhat_mean.squeeze(1)
            self.LOG_xhat_std = self.LOG_xhat_std.squeeze(1)

        elif batch_first:
            self.LOG_x = swap01(self.LOG_x)
            self.LOG_x_mean = swap01(self.LOG_x_mean)
            self.LOG_v = swap01(self.LOG_v)
            self.LOG_I_dec = swap01(self.LOG_I_dec)
            self.LOG_xhat_mean = swap01(self.LOG_xhat_mean)
            self.LOG_xhat_std = swap01(self.LOG_xhat_std)


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
            E = \displaystyle \sum_t \left( \log p(\I_t \mid \x_t) - \KL{q(\x_t \mid \I_t)}{p(\x_t \mid \x_{t-1}, \u_{t-1}; \v_{t})} \right)
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
        * **observation**: tensor of shape :math:`(T, N, C, H, W)`
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

    def forward(self, action: Tensor, observation: Tensor, delta: Tensor):
        """"""

        T = len(action)

        self.init_LOG()

        E_sum: Tensor = 0  # = Nagative ELBO
        E_ll_sum: Tensor = 0  # Not use for training
        E_kl_sum: Tensor = 0  # Not use for training

        for t in range(T):
            u_tn1, I_t = action[t], observation[t]

            _, B, D = action.shape  # _ : T

            if t == 0:
                v_t = torch.zeros(size=(B, D), device=action.device, dtype=action.dtype)
                x_t = self.cell.q_encoder.given(I_t).rsample()
                I_dec = self.cell.p_decoder.given(x_t).decode()

                x_tn1 = x_t
                v_tn1 = v_t

            else:
                output = self.cell(I_t=I_t, x_tn1=x_tn1, u_tn1=u_tn1, v_tn1=v_tn1, dt=delta[t])

                E_sum += output.E
                E_ll_sum += output.E_ll
                E_kl_sum += output.E_kl
                x_tn1 = output.x_t
                v_tn1 = output.v_t
                I_dec = self.cell.p_decoder.decode()

            # ##### DEBUG:
            # print(f"time: {t}")
            # tp_debug.check_dist_model(self.cell)

            if self.is_save:
                self.LOG_x.append(to_np(x_t))
                self.LOG_v.append(to_np(v_t))
                self.LOG_I_dec.append(to_np(I_dec))
                self.LOG_x_mean.append(to_np(self.cell.q_encoder.loc))

        # self.LOG2numpy()

        E = E_sum / T
        E_ll = E_ll_sum / T
        E_kl = E_kl_sum / T

        return E, E_ll, E_kl


class NewtonianVAEDerivation(NewtonianVAEDerivationBase):
    r"""Computes ELBO based on formula (23).

    Computes according to the following formula:

    .. math::
        \begin{array}{ll} \\
            \v_t = \v_{t-1} + \Delta t \cdot (A\x_{t-1} + B\v_{t-1} + C\u_{t-1}) \\
            \x_{t} \sim p(\x_t \mid \x_{t-1}, \u_{t-1}; \v_{t}) \\
            \xhat_{t} \sim p(\xhat_t \mid \x_{t-1}, \u_{t-1}) \\
            E = \displaystyle \sum_t \left( \log p(\I_t \mid \xhat_t) - \KL{q(\x_t \mid \I_t)}{p(\x_t \mid \x_{t-1}, \u_{t-1}; \v_{t})} \right)
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
        * **observation**: tensor of shape :math:`(T, N, C, H, W)`
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

    def forward(self, action: Tensor, observation: Tensor, delta: Tensor):
        """"""

        T = len(action)

        self.init_LOG()

        E_sum: Tensor = 0  # = Nagative ELBO
        E_ll_sum: Tensor = 0  # Not use for training
        E_kl_sum: Tensor = 0  # Not use for training

        for t in range(T):
            u_tn1, I_t = action[t], observation[t]

            _, B, D = action.shape  # _ : T

            if t == 0:
                v_t = torch.zeros(size=(B, D), device=action.device, dtype=action.dtype)
                x_t = self.cell.q_encoder.given(I_t).rsample()
                I_dec = torch.full_like(I_t, torch.nan)

                x_tn1 = x_t
                v_tn1 = v_t

                if self.is_save:
                    self.LOG_xhat_mean.append(np.full((B, self.cell.dim_xhat), np.nan))
                    self.LOG_xhat_std.append(np.full((B, self.cell.dim_xhat), np.nan))

            else:
                output = self.cell(I_t=I_t, x_tn1=x_tn1, u_tn1=u_tn1, v_tn1=v_tn1, dt=delta[t])

                E_sum += output.E
                E_ll_sum += output.E_ll
                E_kl_sum += output.E_kl
                x_tn1 = output.x_t
                v_tn1 = output.v_t
                I_dec = self.cell.p_decoder.decode()

                if self.is_save:
                    self.LOG_xhat_mean.append(to_np(self.cell.p_xhat.loc))
                    self.LOG_xhat_std.append(to_np(self.cell.p_xhat.scale))

            # ##### DEBUG:
            # print(f"time: {t}")
            # tp_debug.check_dist_model(self.cell)

            if self.is_save:
                self.LOG_x.append(to_np(x_t))
                self.LOG_v.append(to_np(v_t))
                self.LOG_I_dec.append(to_np(I_dec))
                self.LOG_x_mean.append(to_np(self.cell.q_encoder.loc))

        # self.LOG2numpy()
        assert len(self.LOG_x) == len(self.LOG_xhat_mean)
        assert len(self.LOG_x) == len(self.LOG_xhat_std)

        E = E_sum / T
        E_ll = E_ll_sum / T
        E_kl = E_kl_sum / T

        return E, E_ll, E_kl


class NewtonianVAEV2(NewtonianVAEBase):
    r"""Computes ELBO based on formula (11).

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
            ELBO = \displaystyle \sum_t \left( \log p(\I_t \mid \x_t) - \KL{q(\x_t \mid \I_t)}{p(\x_t \mid \x_{t-1}, \u_{t-1}; \v_{t})} \right)
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
        * **observation**: tensor of shape :math:`(T, N, C, H, W)`
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

    Acknowledgements:
        `Ito <https://github.com/ItoMasaki>`_ helped me understand the detailed formulas.
        It is important to use :math:`\v_{t-1} = (\x_{t-1} - \x_{t-2}) / \Delta t` for :math:`\v_{t-1}` in :math:`\v_t = \v_{t-1} + \Delta t \cdot (A\x_{t-1} + B\v_{t-1} + C\u_{t-1})`.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        self.cell = NewtonianVAEV2Cell(*args, **kwargs)

    def forward(self, action: Tensor, observation: Tensor, delta: Tensor):
        """"""

        T = len(action)

        self.init_LOG()

        E_sum: Tensor = 0  # = Nagative ELBO
        E_ll_sum: Tensor = 0
        E_kl_sum: Tensor = 0

        for t in range(T):
            u_tn1, I_t = action[t], observation[t]

            _, B, D = action.shape  # _ : T

            if t == 0:
                x_q_t = self.cell.q_encoder.given(I_t).rsample()
                v_t = torch.zeros(size=(B, D), device=action.device, dtype=action.dtype)
                I_dec = self.cell.p_decoder.given(x_q_t).decode()

                x_q_tn1 = x_q_t

            elif t == 1:
                x_q_t = self.cell.q_encoder.given(I_t).rsample()
                v_t = (x_q_t - x_q_tn1) / delta[t]
                I_dec = self.cell.p_decoder.given(x_q_t).decode()

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

            x_q_tn2 = x_q_tn1
            x_q_tn1 = x_q_t

            # ##### DEBUG:
            # print(f"time: {t}")
            # tp_debug.check_dist_model(self.cell)

            if self.is_save:
                self.LOG_x.append(to_np(x_q_t))
                self.LOG_v.append(to_np(v_t))
                self.LOG_I_dec.append(to_np(I_dec))
                self.LOG_x_mean.append(to_np(self.cell.q_encoder.loc))

        E = E_sum / T
        E_ll = E_ll_sum / T
        E_kl = E_kl_sum / T

        return E, E_ll, E_kl


class NewtonianVAEV2Derivation(NewtonianVAEDerivationBase):
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
            ELBO = \displaystyle \sum_t \left( \log p(\I_t \mid \xhat_t) - \KL{q(\x_t \mid \I_t)}{p(\x_t \mid \x_{t-1}, \u_{t-1}; \v_{t})} \right)
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
        * **observation**: tensor of shape :math:`(T, N, C, H, W)`
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

    def forward(self, action: Tensor, observation: Tensor, delta: Tensor):
        """"""

        T = len(action)

        self.init_LOG()

        E_sum: Tensor = 0  # = Nagative ELBO
        E_ll_sum: Tensor = 0
        E_kl_sum: Tensor = 0

        for t in range(T):
            u_tn1, I_t = action[t], observation[t]

            _, B, D = action.shape  # _ : T

            if t == 0:
                x_q_t = self.cell.q_encoder.given(I_t).rsample()
                v_t = torch.zeros(size=(B, D), device=action.device, dtype=action.dtype)
                I_dec = torch.full_like(I_t, torch.nan)

                x_q_tn1 = x_q_t

                if self.is_save:
                    self.LOG_xhat_mean.append(np.full((B, self.cell.dim_xhat), np.nan))
                    self.LOG_xhat_std.append(np.full((B, self.cell.dim_xhat), np.nan))

            elif t == 1:
                x_q_t = self.cell.q_encoder.given(I_t).rsample()
                v_t = (x_q_t - x_q_tn1) / delta[t]
                xhat_t = self.cell.p_xhat.given(x_q_tn1, u_tn1).rsample()
                I_dec = self.cell.p_decoder.given(xhat_t).decode()

                if self.is_save:
                    self.LOG_xhat_mean.append(to_np(self.cell.p_xhat.loc))
                    self.LOG_xhat_std.append(to_np(self.cell.p_xhat.scale))

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
                    self.LOG_xhat_mean.append(to_np(self.cell.p_xhat.loc))
                    self.LOG_xhat_std.append(to_np(self.cell.p_xhat.scale))

            x_q_tn2 = x_q_tn1
            x_q_tn1 = x_q_t

            # ##### DEBUG:
            # print(f"time: {t}")
            # tp_debug.check_dist_model(self.cell)

            if self.is_save:
                self.LOG_x.append(to_np(x_q_t))
                self.LOG_v.append(to_np(v_t))
                self.LOG_I_dec.append(to_np(I_dec))
                self.LOG_x_mean.append(to_np(self.cell.q_encoder.loc))

        # self.LOG2numpy()
        assert len(self.LOG_x) == len(self.LOG_xhat_mean)
        assert len(self.LOG_x) == len(self.LOG_xhat_std)

        E = E_sum / T
        E_ll = E_ll_sum / T
        E_kl = E_kl_sum / T

        return E, E_ll, E_kl


NewtonianVAEFamily = Union[
    NewtonianVAE,
    NewtonianVAEDerivation,
    NewtonianVAEV2,
    NewtonianVAEV2Derivation,
]


class MNVAE(NewtonianVAEBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        self.cell = MNVAECell(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> Tuple[Tensor, Dict[str, float]]:
        return super().__call__(*args, **kwargs)

    def forward(self, batchdata: Dict[str, Tensor]):
        """"""

        action = batchdata["action"]
        delta = batchdata["delta"]
        camera0 = batchdata["camera0"]
        camera1 = batchdata["camera1"]

        T = len(action)

        self.init_LOG()

        E_sum: Tensor = 0  # = Nagative ELBO

        image_losses = 0
        KL_loss: Tensor = 0

        for t in range(T):
            u_tn1 = action[t]
            I_t = [camera0[t], camera1[t]]

            _, B, D = action.shape  # _ : T

            if t == 0:
                x_q_t = self.cell.q_encoder.given(I_t).rsample()
                v_t = torch.zeros(size=(B, D), device=action.device, dtype=action.dtype)
                # I_dec = self.cell.p_decoder.given(x_q_t).decode()

                x_q_tn1 = x_q_t

            elif t == 1:
                x_q_t = self.cell.q_encoder.given(I_t).rsample()
                v_t = (x_q_t - x_q_tn1) / delta[t]
                # I_dec = self.cell.p_decoder.given(x_q_t).decode()

            else:
                output = self.cell(
                    I_t=I_t,
                    x_q_tn1=x_q_tn1,
                    x_q_tn2=x_q_tn2,
                    u_tn1=u_tn1,
                    dt=delta[t],
                )

                E_sum += output.E
                image_losses += np.array(output.image_losses)
                KL_loss += output.KL_loss

                x_q_t = output.x_q_t
                v_t = output.v_t
                # I_dec = self.cell.p_decoder.decode()

            x_q_tn2 = x_q_tn1
            x_q_tn1 = x_q_t

            # ##### DEBUG:
            # print(f"time: {t}")
            # tp_debug.check_dist_model(self.cell)

            if self.is_save:
                self.LOG_x.append(to_np(x_q_t))
                self.LOG_v.append(to_np(v_t))
                # self.LOG_I_dec.append(to_np(I_dec))
                self.LOG_x_mean.append(to_np(self.cell.q_encoder.loc))

        E = E_sum / T
        L = -E

        image_losses /= T
        losses = {}
        for i in range(len(image_losses)):
            losses[f"camera{i} Loss"] = image_losses[i]

        losses["KL Loss"] = KL_loss / T

        return L, losses
