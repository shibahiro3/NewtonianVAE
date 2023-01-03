import os
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim

import models.core
import models.pcontrol
import mypython.ai.torchprob as tp
import tool.util
from models.core import NewtonianVAEFamily
from models.pcontrol import PControl
from mypython.ai.util import SequenceDataLoader, print_module_params, reproduce
from mypython.pyutil import RemainingTime, s2dhms_str
from mypython.terminal import Color, Prompt
from tool import paramsmanager
from tool.util import Preferences


def preprocess_x(x, coef=1):
    return x * coef


def preprocess_u(u, coef=0.2):
    return u * coef


def postprocess_x(x, coef=1):
    return x / coef


def postprocess_u(u, coef=0.2):
    return u / coef


def train(
    config: str,
    config_ctrl: str,
):
    torch.set_grad_enabled(True)

    params_ctrl = paramsmanager.Params(config_ctrl)

    if params_ctrl.train.seed is None:
        params_ctrl.train.seed = np.random.randint(0, 2**16)
    reproduce(params_ctrl.train.seed)

    dtype, device = tool.util.dtype_device(
        dtype=params_ctrl.train.dtype,
        device=params_ctrl.train.device,
    )

    trainloader = SequenceDataLoader(
        root=Path(params_ctrl.path.data_dir, "episodes"),
        names=["action", "observation"],
        start=params_ctrl.train.data_start,
        stop=params_ctrl.train.data_stop,
        batch_size=params_ctrl.train.batch_size,
        dtype=dtype,
        device=device,
    )
    # action, observation = next(trainloader)

    model, _, weight_path, _ = tool.util.load(
        root=paramsmanager.Params(config).path.saves_dir,
        model_place=models.core,
    )
    model: NewtonianVAEFamily
    model.type(dtype)
    model.to(device)
    model.eval()

    p_pctrl, managed_dir, weight_dir, resume_weight_path = tool.util.creator(
        root=params_ctrl.path.saves_dir,
        model_place=models.pcontrol,
        model_name=params_ctrl.model,
        model_params=params_ctrl.model_params,
    )
    p_pctrl: PControl
    p_pctrl.type(dtype)
    p_pctrl.to(device)
    p_pctrl.train()
    # print_module_params(p_pctrl)
    optimizer = optim.Adam(p_pctrl.parameters(), params_ctrl.train.learning_rate)

    params_ctrl.path.used_nvae_weight = weight_path
    params_ctrl.pid = os.getpid()
    params_ctrl.save(Path(managed_dir, "params_saved.json5"))

    record_Loss = []

    Preferences.put(managed_dir, "running", True)

    def end_process():
        Preferences.remove(managed_dir, "running")

        if len(list(weight_dir.glob("*"))) > 0:
            np.save(Path(managed_dir, "LOG_Loss.npy"), record_Loss)
        else:
            shutil.rmtree(managed_dir)

        print("\nEnd of train")

    time_prev = time.perf_counter()

    try:
        tp.config.check_value = params_ctrl.train.check_value  # if False, A little bit faster
        remaining = RemainingTime(max=params_ctrl.train.epochs * len(trainloader), size=50)
        for epoch in range(1, params_ctrl.train.epochs + 1):
            for action, observation in trainloader:

                action = action.reshape((-1,) + action.shape[-1:])
                observation = observation.reshape((-1,) + observation.shape[-3:])

                u_t = action
                x_t = model.cell.q_encoder.given(observation).sample()

                # print("=================")
                # print(u_t.shape)
                # print(x_t.shape)
                # print(u_t.min().item(), u_t.max().item())
                # print(x_t.min().item(), x_t.max().item())

                u_t = preprocess_u(u_t)
                x_t = preprocess_x(x_t)

                L = -tp.log(p_pctrl, u_t).given(x_t).mean()

                optimizer.zero_grad()
                L.backward()
                optimizer.step()
                record_Loss.append(L.item())

                remaining.update()
                # Enable ctrl + click on paths in vscode ternimal
                # If I print it as it is, the terminal updates too fast to click on it.
                time_now = time.perf_counter()
                if time_now - time_prev > 0.5:
                    Prompt.print_one_line(
                        (
                            f"Epoch {epoch} | "
                            f"Loss: {L:.4f} | "
                            f"Elapsed: {s2dhms_str(remaining.elapsed)} | "
                            f"Remaining: {s2dhms_str(remaining.time)} | "
                            f"ETA: {remaining.eta} "
                        )
                    )
                    time_prev = time_now

            if epoch % params_ctrl.train.save_per_epoch == 0:
                torch.save(p_pctrl.state_dict(), Path(weight_dir, f"{epoch}.pth"))
                Prompt.print_one_line(
                    (
                        f"Epoch {epoch} | "
                        f"Loss: {L:.4f} | "
                        f"Elapsed: {s2dhms_str(remaining.elapsed)} | "
                        f"Remaining: {s2dhms_str(remaining.time)} | "
                        f"ETA: {remaining.eta} "
                    )
                )
                print("saved")

        plt.title("Loss")
        plt.plot(record_Loss)
        plt.show()

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt")
    except:
        print("=== traceback ===")
        print(traceback.format_exc())

    end_process()
