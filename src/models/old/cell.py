class CellWrap:
    """Deprecated"""

    def __init__(self, cell) -> None:
        self.cell = cell

        self.reset()

    def step(self, action: Tensor, observation: Tensor):
        """
        action: u_tn1
        observation: I_t
        """

        if observation.ndim == 3:
            observation = observation.unsqueeze(0)

        x_t = self.cell.q_encoder.given(observation).sample()

        if hasattr(self.cell, "dim_xhat"):
            if self.x_tn1 is None:
                I_t_dec = torch.full_like(observation, torch.nan)
            else:
                xhat_t = self.cell.p_xhat.given(self.x_tn1, action).sample()
                I_t_dec = self.cell.p_decoder.given(xhat_t).decode()

        else:
            I_t_dec = self.cell.p_decoder.given(x_t).decode()

        self.x_tn1 = x_t

        return x_t, I_t_dec

    def reset(self):
        self.x_tn1: Tensor = None


class NewtonianVAECell(NewtonianVAECellBase):
    """
    Eq (11)
    """

    def __init__(
        self,
        dim_x: int,
        regularization: bool,
        velocity: dict,
        transition: dict,
        encoder: dict,
        decoder: dict,
    ) -> None:
        super().__init__(dim_x=dim_x, regularization=regularization)

        self.f_velocity = component.Velocity(**velocity)
        self.p_transition = component.Transition(**transition)
        self.q_encoder = component.Encoder(**encoder)
        self.p_decoder = component.Decoder(**decoder)

    @dataclasses.dataclass
    class Pack:
        E: Tensor  # Use for training
        E_ll: Tensor
        E_kl: Tensor
        x_t: Tensor  # Use for training
        v_t: Tensor  # Use for training

    def __call__(self, *args, **kwargs) -> Pack:
        return super().__call__(*args, **kwargs)

    def forward(self, I_t: Tensor, x_tn1: Tensor, u_tn1: Tensor, v_tn1: Tensor, dt: Tensor):
        """"""

        if self.training or self.force_training:
            v_t = self.f_velocity(x_tn1, u_tn1, v_tn1, dt)
            x_t = self.p_transition.given(x_tn1, v_t, dt).rsample()
            E_ll = self.img_reduction(tp.log(self.p_decoder, I_t).given(x_t))
            E_kl = self.vec_reduction(tp.KLdiv(self.q_encoder.given(I_t), self.p_transition))
            E = E_ll - self.kl_beta * E_kl

            if self.regularization:
                E -= self.vec_reduction(tp.KLdiv(self.q_encoder, tp.Normal01))

            return self.Pack(E=E, E_ll=E_ll.detach(), E_kl=E_kl.detach(), x_t=x_t, v_t=v_t)

        else:
            x_t = self.q_encoder.given(I_t).rsample()
            self.p_decoder.given(x_t)  # for cell.p_decoder.decode()
            v_t = (x_t - x_tn1) / dt  # for only visualize
            return self.Pack(E=0, E_ll=0, E_kl=0, x_t=x_t, v_t=v_t)


class NewtonianVAEDerivationCell(NewtonianVAECellBase):
    """
    Eq (23)
    """

    def __init__(
        self,
        dim_x: int,
        dim_xhat: int,
        regularization: bool,
        velocity: dict,
        transition: dict,
        encoder: dict,
        decoder: dict,
        pxhat: dict,
    ) -> None:
        super().__init__(dim_x=dim_x, regularization=regularization)

        self.dim_xhat = dim_xhat

        self.f_velocity = component.Velocity(**velocity)
        self.p_transition = component.Transition(**transition)
        self.q_encoder = component.Encoder(**encoder)
        self.p_decoder = component.Decoder(**decoder)  # p(I_t | xhat_t)

        # p(xhat_t | x_{t-1}, u_{t-1})
        self.p_xhat = component.Pxhat(**pxhat)

    @dataclasses.dataclass
    class Pack:
        E: Tensor  # Use for training
        E_ll: Tensor
        E_kl: Tensor
        x_t: Tensor  # Use for training
        v_t: Tensor  # Use for training

    def __call__(self, *args, **kwargs) -> Pack:
        return super().__call__(*args, **kwargs)

    def forward(self, I_t: Tensor, x_tn1: Tensor, u_tn1: Tensor, v_tn1: Tensor, dt: Tensor):
        """"""

        if self.training:
            v_t = self.f_velocity(x_tn1, u_tn1, v_tn1, dt)
            xhat_t = self.p_xhat.given(x_tn1, u_tn1).rsample()
            E_ll = self.img_reduction(tp.log(self.p_decoder, I_t).given(xhat_t))
            E_kl = self.vec_reduction(
                tp.KLdiv(self.q_encoder.given(I_t), self.p_transition.given(x_tn1, v_t, dt))
            )
            E = E_ll - self.kl_beta * E_kl

            if self.regularization:
                E -= self.vec_reduction(tp.KLdiv(self.q_encoder, tp.Normal01))

            x_t = self.q_encoder.rsample()

            return self.Pack(
                E=E,
                E_ll=E_ll.detach(),
                E_kl=E_kl.detach(),
                x_t=x_t,
                v_t=v_t,
            )

        else:
            x_t = self.q_encoder.given(I_t).rsample()
            xhat_t = self.p_xhat.given(x_tn1, u_tn1).rsample()
            self.p_decoder.given(xhat_t)  # for cell.p_decoder.decode()
            v_t = (x_t - x_tn1) / dt  # for only visualize
            return self.Pack(E=0, E_ll=0, E_kl=0, x_t=x_t, v_t=v_t)


class NewtonianVAEV3Cell(nn.Module):
    def __init__(
        self,
        dim_x: int,
        velocity,
        transition,
        encoder,
        decoder,
        regularization: bool = False,
    ) -> None:
        super().__init__()

        self.dim_x = dim_x
        self.regularization = regularization

        self.kl_beta = 1
        self.force_training = False  # for add_graph of tensorboard

        self.f_velocity = component.Velocity(**velocity)
        self.p_transition = component.Transition(**transition)
        self.q_encoder = component.Encoder(**encoder)
        self.p_decoder = component.Decoder(**decoder)


class NewtonianVAEV4Cell(NewtonianVAECell):
    def __init__(
        self,
        *,
        dim_x: int,
        regularization: bool,
        velocity: dict,
        transition: dict,
        encoder: dict,
        decoder: dict,
        pre_state_dict: Optional[OrderedDict] = None,
    ):
        super().__init__(dim_x=dim_x, regularization=regularization)

        self.f_velocity = component.Velocity(dim_x=dim_x, **velocity)
        self.p_transition = component.Transition(**transition)
        self.q_encoder = component.Encoder(
            dim_x=dim_x,
            pre_state_dict=pre_state_dict,
            **encoder,
        )
        self.p_decoder = component.Decoder(
            dim_input=dim_x,
            pre_state_dict=pre_state_dict,
            **decoder,
        )


class NewtonianVAEV2DerivationCell(NewtonianVAECellBase):
    """
    Eq (23)
    """

    def __init__(
        self,
        dim_x: int,
        dim_xhat: int,
        velocity: dict,
        transition: dict,
        encoder: dict,
        decoder: dict,
        pxhat: dict,
        regularization: bool = False,
    ) -> None:
        super().__init__(dim_x=dim_x, regularization=regularization)

        self.f_velocity = component.Velocity(**velocity)
        self.p_transition = component.Transition(**transition)
        self.q_encoder = component.Encoder(**encoder)
        self.p_decoder = component.Decoder(**decoder)  # p(I_t | xhat_t)

        # p(xhat_t | x_{t-1}, u_{t-1})
        self.p_xhat = component.Pxhat(**pxhat)

        self.dim_xhat = dim_xhat

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
        xhat_t = self.p_xhat.given(x_q_tn1, u_tn1).rsample()
        E_ll = self.img_reduction(tp.log(self.p_decoder, I_t).given(xhat_t))
        E_kl = self.vec_reduction(
            tp.KLdiv(self.q_encoder.given(I_t), self.p_transition.given(x_q_tn1, v_t, dt))
        )
        E = E_ll - self.kl_beta * E_kl

        if self.regularization:
            E -= self.vec_reduction(tp.KLdiv(self.q_encoder, tp.Normal01))

        x_q_t = self.q_encoder.rsample()

        return self.Pack(
            E=E,
            x_q_t=x_q_t,
            E_ll=E_ll.detach(),
            E_kl=E_kl.detach(),
            v_t=v_t.detach(),
        )


class NVAEDecoderFreeCell(NewtonianVAECellBase):
    def __init__(
        self,
        *,
        dim_x: int,
        velocity: dict,
        transition: dict,
        encoder: dict,
        regularization: bool = False,
    ):
        super().__init__(dim_x=dim_x, regularization=regularization)

        self.f_velocity = component.Velocity(dim_x=dim_x, **velocity)
        self.p_transition = component.Transition(**transition)
        self.q_encoder = component.Encoder(dim_x=dim_x, **encoder)

    @dataclasses.dataclass
    class Pack:
        E: Tensor  # Use for training
        E_kl: Tensor
        beta_kl: Tensor
        x_q_t: Tensor  # Use for training
        v_t: Tensor

    def __call__(self, *args, **kwargs) -> Pack:
        return super().__call__(*args, **kwargs)

    def forward(self, I_t: Tensor, x_q_tn1: Tensor, x_q_tn2: Tensor, u_tn1: Tensor, dt: Tensor):
        """"""
        v_tn1 = (x_q_tn1 - x_q_tn2) / dt
        v_t = self.f_velocity(x_q_tn1, u_tn1, v_tn1, dt)

        E_kl = self.vec_reduction(
            tp.KLdiv(self.q_encoder.given(I_t), self.p_transition.given(x_q_tn1, v_t, dt))
        )
        beta_kl = self.kl_beta * E_kl
        E = -beta_kl

        if self.regularization:
            E -= self.vec_reduction(tp.KLdiv(self.q_encoder, tp.Normal01))

        x_q_t = self.q_encoder.rsample()

        return self.Pack(
            E=E,
            x_q_t=x_q_t,
            E_kl=E_kl.detach(),
            beta_kl=beta_kl.detach(),
            v_t=v_t.detach(),
        )
