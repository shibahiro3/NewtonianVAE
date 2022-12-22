import os
import sys

sys.path.append(os.pardir)


import shutil
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim

import models.core
import mypython.ai.torchprob as tp
import tool.util
from models.cell import CellWrap
from models.core import NewtonianVAECell, NewtonianVAEDerivationCell, NewtonianVAEFamily
from models.pcontrol import PControl
from mypython.ai.torchprob.distributions.gmm import log_gmm
from mypython.ai.util import SequenceDataLoader, print_module_params, reproduce
from mypython.pyutil import s2dhms_str
from mypython.terminal import Color, Prompt
from tool import paramsmanager
from tool.util import dtype_device


def train(
    config: str,
    path_model: str,
):
    torch.set_grad_enabled(True)

    dim_x = 2
    K = 3
    T = 100
    BS = 10

    params = paramsmanager.Params(config)

    if params.train.seed is None:
        params.train.seed = np.random.randint(0, 2**16)
    reproduce(params.train.seed)

    dtype, device = dtype_device(
        dtype=params.train.dtype,
        device=params.train.device,
    )

    trainloader = SequenceDataLoader(
        root=Path(params.external.data_path, "episodes"),
        names=["action", "observation"],
        start=params.train.data_start,
        stop=params.train.data_stop,
        batch_size=params.train.batch_size,
        dtype=dtype,
        device=device,
    )

    model, manage_dir, weight_path, _params = tool.util.load(
        root=path_model, model_place=models.core
    )

    model: NewtonianVAEFamily
    model.type(dtype)
    model.to(device)
    model.eval()

    p_pctrl = PControl(
        N=3,
        dim_x=dim_x,
        dim_x_goal_middle=5,
        dim_pi_middle=4,
        dim_K_middle=5,
        std=0.001,
    )
    p_pctrl.type(dtype)
    p_pctrl.to(device)
    p_pctrl.train()
    # print_module_params(p_pctrl)

    optimizer = optim.Adam(p_pctrl.parameters(), params.train.learning_rate)

    record_Loss = []

    for epoch in range(1, params.train.epochs + 1):
        cell_wrap = CellWrap(cell=model.cell)
        for action, observation in trainloader:

            action = action.reshape((-1,) + action.shape[-1:])
            observation = observation.reshape((-1,) + observation.shape[-3:])

            # print("=================")
            # print(action.shape)
            # print(observation.shape)

            u_t = action.detach()
            x_t, I_t_dec = cell_wrap.step(action=action, observation=observation)
            x_t.detach()

            L = -tp.log(p_pctrl, u_t, x_t).mean()

            optimizer.zero_grad()
            L.backward()
            optimizer.step()

            Prompt.print_one_line(f"Epoch {epoch} | Loss: {L:.4f} ")
            record_Loss.append(L.item())

    print()

    plt.title("Loss")
    plt.plot(record_Loss)
    plt.show()


if __name__ == "__main__":
    train()
