class NewtonianVAE(BaseWithCache):
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

        self.cell = cell.NewtonianVAECell(*args, **kwargs)

    def forward(self, action: Tensor, observation: Tensor, delta: Tensor):
        """"""

        T = len(action)

        self.init_cache()

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


class NewtonianVAEDerivation(BaseWithCache):
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

        self.cell = cell.NewtonianVAEDerivationCell(*args, **kwargs)

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

        E = E_sum / T
        E_ll = E_ll_sum / T
        E_kl = E_kl_sum / T

        return E, E_ll, E_kl


class NewtonianVAEV2Derivation(BaseWithCache):
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

        self.cell = cell.NewtonianVAEV2DerivationCell(*args, **kwargs)

    def forward(self, action: Tensor, observation: Tensor, delta: Tensor):
        """"""

        T = len(action)

        self.init_cache()

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

        E = E_sum / T
        E_ll = E_ll_sum / T
        E_kl = E_kl_sum / T

        return E, E_ll, E_kl
