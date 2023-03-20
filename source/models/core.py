"""
x_t    == x_t
x_tn1  == x_{t-1}
x_tn2  == x_{t-2}
x_tp1  == x_{t+1}
x_tp2  == x_{t+2}
"""

import dataclasses
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import NumberType, Tensor, nn

import mypython.ai.torchprob as tp
import mypython.ai.torchprob.debug as tp_debug
from mypython import rdict
from mypython.ai.util import find_function, swap01, to_np
from mypython.terminal import Color

from . import cell

CacheType = Dict[str, Union[list, Tensor, np.ndarray]]


class BaseWithCache(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.is_save = False

        self._cache: CacheType = {}

    def __call__(self, *args, **kwargs) -> Tuple[Tensor, Dict[str, float]]:
        """
        Returns:
            Loss (Tensor), info (Dict)
        """
        return super().__call__(*args, **kwargs)

    @property
    def cache(self):
        return self._cache

    def init_cache(self) -> None:
        self._cache = {}

    def convert_cache(self, type_to="list", treat_batch=None, verbose=False) -> CacheType:
        """
        return_type: "list" or "torch" or "numpy"
        treat_batch: None or "first" or "squeeze"
            None : (T, B, D)
            "first" : (B, T, D)
            "squeeze" : (T, D)  Valid only when batch size is 1
        """

        if type_to == "list":
            if verbose:
                rdict.show(self._cache, "cache")
            return self._cache

        if type_to == "torch":
            rdict.to_torch(self._cache)
        elif type_to == "numpy":
            rdict.to_numpy(self._cache)
        else:
            assert False

        if treat_batch is None:
            pass
        elif treat_batch == "first":
            rdict.apply_(self._cache, swap01)
        elif treat_batch == "squeeze":
            rdict.apply_(self._cache, lambda x: x.squeeze(1))
        else:
            assert False

        if verbose:
            rdict.show(self._cache, "cache")
        return self._cache


class NewtonianVAEBase(BaseWithCache):
    def __init__(self) -> None:
        super().__init__()

        self.dim_x = 0

    @property
    def camera_names(self) -> List[str]:
        raise NotImplementedError()

    def encode(self, I_t) -> Optional[Tensor]:
        """return "position" """
        return None

    def decode(self, x_t) -> Optional[Dict[str, Tensor]]:
        """return reconstructed image"""
        return None


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

    def __init__(
        self,
        *,
        camera_name: str,
        dim_x: int,
        regularization: int,
        velocity: dict,
        transition: dict,
        encoder: dict,
        decoder: dict,
    ) -> None:
        super().__init__()

        self.camera_name = camera_name

        self.cell = cell.NewtonianVAEV2Cell(
            dim_x=dim_x,
            regularization=regularization,
            velocity=velocity,
            transition=transition,
            encoder=encoder,
            decoder=decoder,
        )

    def forward(self, batchdata: Dict[str, Tensor]):
        """"""

        action = batchdata["action"]
        delta = batchdata["delta"]

        T = len(action)

        self.init_cache()

        E_sum: Tensor = 0  # = Nagative ELBO
        E_ll_sum: Tensor = 0
        E_kl_sum: Tensor = 0
        beta_kl: Tensor = 0

        for t in range(T):
            u_tn1 = action[t]
            I_t = batchdata["camera"][self.camera_name][t]

            _, B, D = action.shape  # _ : T

            if t == 0:
                x_q_t = self.cell.q_encoder.given(I_t).rsample()
                v_t = torch.zeros(size=(B, D), device=action.device, dtype=action.dtype)
                I_dec = self.cell.p_decoder(x_q_t)

                x_q_tn1 = x_q_t

            elif t == 1:
                x_q_t = self.cell.q_encoder.given(I_t).rsample()
                v_t = (x_q_t - x_q_tn1) / delta[t]
                I_dec = self.cell.p_decoder(x_q_t)

            else:
                output = self.cell(
                    I_t=I_t, x_q_tn1=x_q_tn1, x_q_tn2=x_q_tn2, u_tn1=u_tn1, dt=delta[t]
                )

                E_sum += output.E
                E_ll_sum += output.E_ll
                E_kl_sum += output.E_kl
                beta_kl += output.beta_kl

                x_q_t = output.x_q_t
                v_t = output.v_t
                I_dec = output.I_t_rec

            x_q_tn2 = x_q_tn1
            x_q_tn1 = x_q_t

            # ##### DEBUG:
            # print(f"time: {t}")
            # tp_debug.check_dist_model(self.cell)

            if self.is_save:
                self.cache["x"].append(x_q_t)
                self.cache["x_mean"].append(self.cell.q_encoder.loc)
                self.cache["x_std"].append(self.cell.q_encoder.scale)
                self.cache["v"].append(v_t)
                self.cache["camera"][self.camera_name].append(I_dec)

        E = E_sum / T
        L = -E

        losses = {
            f"{self.camera_name} Loss": -E_ll_sum / T,
            "KL Loss": E_kl_sum / T,
            "Beta KL Loss": beta_kl / T,
        }
        return L, losses

    def init_cache(self):
        super().init_cache()
        self.cache["x"] = []
        self.cache["x_mean"] = []
        self.cache["x_std"] = []
        self.cache["v"] = []

        self.cache["camera"] = {}
        self.cache["camera"][self.camera_name] = []

    @property
    def camera_names(self) -> List[str]:
        return [self.camera_names]

    def encode(self, I_t) -> Tensor:
        return self.cell.q_encoder.given(I_t).rsample()

    def decode(self, x_t) -> Dict[str, Dict[str, Tensor]]:
        return {self.camera_name: self.cell.p_decoder(x_t)}


class NewtonianVAEV3(NewtonianVAEBase):
    """Execution speed was not faster"""

    def __init__(self, *args, camera_name: str, **kwargs) -> None:
        super().__init__()

        self.cell = cell.NewtonianVAEV3Cell(*args, **kwargs)
        self.camera_name = camera_name

    def forward(self, batchdata: Dict[str, Tensor]):
        """"""

        action = batchdata["action"]
        d = batchdata["delta"]

        T, B, D = action.shape

        self.init_cache()

        I = batchdata["camera"][self.camera_name]  # (T, B, C, H, W)
        T, B, C, H, W = I.shape
        I_ = I.reshape(-1, C, H, W)
        q_enc = self.cell.q_encoder.given(I_)
        x_q_loc = q_enc.loc.reshape(T, B, -1)
        x_q_scale = q_enc.scale.reshape(T, B, -1)
        x_q = q_enc.rsample().reshape(T, B, -1)

        if self.is_save:
            self.cache["x"] = x_q
            self.cache["x_mean"] = x_q_loc
            self.cache["x_std"] = x_q_scale

        v_t = torch.zeros(size=(B, D), device=action.device, dtype=action.dtype)
        x_p = []

        E_kl = 0
        for t in range(2, T):
            u_tn1 = action[t]
            v_tn1 = (x_q[t - 1] - x_q[t - 2]) / d[t]
            v_t = self.cell.f_velocity(x_q[t - 1], u_tn1, v_tn1, d[t])
            x_p_t = self.cell.p_transition.given(x_q[t - 1], v_t, d[t]).rsample()
            x_p.append(x_p_t)

            E_kl += cell.NewtonianVAECellBase.vec_reduction(
                tp.functions.KL_normal_normal(
                    x_q_loc[t],
                    x_q_scale[t],
                    self.cell.p_transition.loc,
                    self.cell.p_transition.scale,
                )
            )

            # if self.cell.regularization:

            if self.is_save:
                self.cache["v"].append(v_t)

        # x_p = torch.stack(x_p).reshape((T-2) * B, -1)
        x_p = torch.stack(x_p).reshape(-1, D)

        I_rec = self.cell.p_decoder(x_p)

        # log Normal Dist.: -0.5 * (((x - mu) / sigma) ** 2 ...
        E_ll = -F.mse_loss(
            I_rec, I[2:].reshape(-1, C, H, W), reduction="none"
        )  # ((T-2)*B, C, H, W)
        E_ll = E_ll.reshape(T - 2, B, C, H, W)
        E_ll = cell.NewtonianVAECellBase.img_reduction(E_ll).sum()

        # if self.cell.regularization:
        #     E -= NewtonianVAECellBase.vec_reduction(tp.KLdiv(self.cell.q_encoder, tp.Normal01))

        beta_kl = self.cell.kl_beta * E_kl
        E = E_ll - beta_kl
        L = -E / T

        if self.is_save:
            I0 = self.cell.p_decoder(x_q[0]).unsqueeze(0)
            I1 = self.cell.p_decoder(x_q[1]).unsqueeze(0)
            I_rec_ = torch.cat([I0, I1, I_rec.reshape(T - 2, B, C, H, W)], dim=0)  # (T, B, C, H, W)
            self.cache["camera"][self.camera_name] = I_rec_

        losses = {
            f"{self.camera_name} Loss": -E_ll / T,
            "KL Loss": E_kl / T,
            "Beta KL Loss": beta_kl / T,
        }
        return L, losses

    def init_cache(self):
        super().init_cache()
        self.cache["x"] = []
        self.cache["x_mean"] = []
        self.cache["x_std"] = []
        self.cache["v"] = []

        self.cache["camera"] = {}
        self.cache["camera"][self.camera_name] = []


class NewtonianvVAEV4(NewtonianVAEV2):
    def __init__(
        self,
        *,
        camera_name: str,
        dim_x: int,
        regularization: int,
        velocity: dict,
        transition: dict,
        encoder: dict,
        decoder: dict,
        pre_weght_path: str = None,
        pre_mode: str = "none",
    ) -> None:
        super().__init__(
            camera_name=camera_name,
            dim_x=dim_x,
            regularization=regularization,
            velocity=velocity,
            transition=transition,
            encoder=encoder,
            decoder=decoder,
        )

        pre_state_dict = None

        if pre_mode != "none":
            assert type(pre_weght_path) == str

            pre_state_dict = torch.load(pre_weght_path)

        self.cell = cell.NewtonianVAEV4Cell(pre_state_dict)


class NVAEDecoderFree(BaseWithCache):
    def __init__(
        self,
        *,
        camera_name: str,
        dim_x: int,
        regularization: int,
        velocity: dict,
        transition: dict,
        encoder: dict,
    ) -> None:
        super().__init__()

        self.camera_name = camera_name

        self.cell = cell.NVAEDecoderFreeCell(
            dim_x=dim_x,
            regularization=regularization,
            velocity=velocity,
            transition=transition,
            encoder=encoder,
        )

    def forward(self, batchdata: Dict[str, Tensor]):
        """"""

        action = batchdata["action"]
        delta = batchdata["delta"]

        T = len(action)

        self.init_cache()

        E_sum: Tensor = 0  # = Nagative ELBO
        E_kl_sum: Tensor = 0
        beta_kl: Tensor = 0

        for t in range(T):
            u_tn1 = action[t]
            I_t = batchdata["camera"][self.camera_name][t]

            _, B, D = action.shape  # _ : T

            if t == 0:
                x_q_t = self.cell.q_encoder.given(I_t).rsample()
                v_t = torch.zeros(size=(B, D), device=action.device, dtype=action.dtype)

                x_q_tn1 = x_q_t

            elif t == 1:
                x_q_t = self.cell.q_encoder.given(I_t).rsample()
                v_t = (x_q_t - x_q_tn1) / delta[t]

            else:
                output = self.cell(
                    I_t=I_t, x_q_tn1=x_q_tn1, x_q_tn2=x_q_tn2, u_tn1=u_tn1, dt=delta[t]
                )

                E_sum += output.E

                E_kl_sum += output.E_kl
                beta_kl += output.beta_kl

                x_q_t = output.x_q_t
                v_t = output.v_t

            x_q_tn2 = x_q_tn1
            x_q_tn1 = x_q_t

            # ##### DEBUG:
            # print(f"time: {t}")
            # tp_debug.check_dist_model(self.cell)

            if self.is_save:
                self.cache["x"].append(x_q_t)
                self.cache["x_mean"].append(self.cell.q_encoder.loc)
                self.cache["x_std"].append(self.cell.q_encoder.scale)
                self.cache["v"].append(v_t)

        E = E_sum / T
        L = -E

        losses = {
            "KL Loss": E_kl_sum / T,
            "Beta KL Loss": beta_kl / T,
        }
        return L, losses

    def init_cache(self):
        super().init_cache()
        self.cache["x"] = []
        self.cache["x_mean"] = []
        self.cache["x_std"] = []
        self.cache["v"] = []


class MNVAE(NewtonianVAEBase):
    def __init__(self, *args, camera_names: list, **kwargs) -> None:
        super().__init__()

        self._camera_names = camera_names

        self.cell = cell.MNVAECell(*args, **kwargs)
        self.dim_x = self.cell.dim_x

    def __call__(self, *args, **kwargs) -> Tuple[Tensor, Dict[str, float]]:
        return super().__call__(*args, **kwargs)

    def forward(self, batchdata: Dict[str, Tensor]):
        """"""

        action = batchdata["action"]
        delta = batchdata["delta"]

        T = len(action)

        self.init_cache()

        E_sum: Tensor = 0  # = Nagative ELBO

        image_losses = 0
        KL_loss: Tensor = 0
        beta_kl: Tensor = 0

        for t in range(T):
            u_tn1 = action[t]
            I_t = [batchdata["camera"][name][t] for name in self._camera_names]

            _, B, D = action.shape  # _ : T

            if t == 0:
                x_q_t = self.cell.q_encoder.given(I_t).rsample()
                v_t = torch.zeros(size=(B, D), device=action.device, dtype=action.dtype)
                I_t_recs = self.cell.p_decoder(x_q_t)

                x_q_tn1 = x_q_t

            elif t == 1:
                x_q_t = self.cell.q_encoder.given(I_t).rsample()
                v_t = (x_q_t - x_q_tn1) / delta[t]
                I_t_recs = self.cell.p_decoder(x_q_t)

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
                beta_kl += output.beta_kl

                x_q_t = output.x_q_t
                v_t = output.v_t
                I_t_recs = output.I_t_recs

            x_q_tn2 = x_q_tn1
            x_q_tn1 = x_q_t

            # ##### DEBUG:
            # print(f"time: {t}")
            # tp_debug.check_dist_model(self.cell)

            if self.is_save:
                self.cache["x"].append(x_q_t)
                self.cache["x_mean"].append(self.cell.q_encoder.loc)
                self.cache["x_std"].append(self.cell.q_encoder.scale)
                self.cache["v"].append(v_t)
                for i, name in enumerate(self._camera_names):
                    self.cache["camera"][name].append(I_t_recs[i])

        E = E_sum / T
        L = -E

        image_losses /= T

        losses = {}
        for i, k in enumerate(self._camera_names):
            losses[f"camera {k} Loss"] = image_losses[i]

        losses.update(
            {
                "KL Loss": KL_loss / T,
                "Beta KL Loss": beta_kl / T,
            }
        )
        return L, losses

    def init_cache(self):
        super().init_cache()
        self.cache["x"] = []
        self.cache["x_mean"] = []
        self.cache["x_std"] = []
        self.cache["v"] = []

        self.cache["camera"] = {}
        for name in self._camera_names:
            self.cache["camera"][name] = []

    @property
    def camera_names(self):
        return self._camera_names

    def encode(self, I_t: List[Tensor]) -> Tensor:
        return self.cell.q_encoder.given(I_t).rsample()

    def decode(self, x_t: Tensor) -> Dict[str, Tensor]:
        I_t_recs = self.cell.p_decoder(x_t)
        dec = {}
        for i, name in enumerate(self._camera_names):
            dec[name] = I_t_recs[i]
        return dec
