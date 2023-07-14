#!/usr/bin/env python3


import os
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import models.controller
import models.core
import mypython.ai.torchprob as tp
import numpy as np
import tool.util
import torch
from models import controller
from models.controller import PControl
from models.core import NewtonianVAEBase
from mypython.ai.util import SequenceDataLoader, print_module_params, reproduce
from mypython.numeric import RemainingTime
from mypython.pyutil import s2dhms_str
from mypython.terminal import Color, Prompt
from tool import paramsmanager
from tool.util import Preferences, create_prepostprocess, creator, dtype_device
from torch import nn, optim


def main():
    train(sys.argv[1], sys.argv[2])


def train(
    config: str,
    config_ctrl: str,
):
    torch.set_grad_enabled(True)

    params = paramsmanager.Params(config_ctrl)

    if params.train.seed is None:
        params.train.seed = np.random.randint(0, 2**16 - 1)
    reproduce(params.train.seed)

    dtype, device = dtype_device(
        dtype=params.train.dtype,
        device=params.train.device,
    )

    preprocess, postprocesses = create_prepostprocess(params, device=device)
    keypaths = params.others.get("keypaths", None)

    trainloader = SequenceDataLoader(
        patterns=params.train.path,
        batch_size=params.train.batch_size,
        max_time=params.train.max_time_length,
        dtype=dtype,
        device=device,
        preprocess=preprocess,
        keypaths=keypaths,
        load_all=params.train.load_all,
    )
    trainloader.sample_batch(verbose=True, print_name="sample train batch")

    model, _, weight_path, _ = tool.util.load(
        root=paramsmanager.Params(config).path.saves_dir,
        model_place=models.core,
    )
    model: NewtonianVAEBase
    model.type(dtype)
    model.to(device)
    model.eval()

    p_pctrl, managed_dir, weight_dir, resume_weight_path = tool.util.creator(
        root=params.path.saves_dir,
        model_place=models.controller,
        model_name=params.model,
        model_params=params.model_params,
    )
    p_pctrl: PControl
    p_pctrl.type(dtype)
    p_pctrl.to(device)
    p_pctrl.train()
    # print_module_params(p_pctrl)
    optimizer = optim.Adam(p_pctrl.parameters(), params.train.learning_rate)

    params.path.used_nvae_weight = weight_path
    params.pid = os.getpid()
    params.save_train_ctrl(Path(managed_dir, "params_saved.json5"))

    def end_process():
        pass

    time_prev = time.perf_counter()

    record_Loss = []

    # s_u = params.preprocess.scale_u
    # if s_u is not None:
    #     scaler_u = tp.Scaler(*s_u)
    # else:
    #     scaler_u = tp.Scaler()

    # s_x = params.preprocess.scale_x
    # if s_x is not None:
    #     scaler_x = tp.Scaler(*s_x)
    # else:
    #     scaler_x = tp.Scaler()

    try:
        tp.config.check_value = params.train.check_value  # if False, A little bit faster
        remaining = RemainingTime(max=params.train.epochs * len(trainloader), size=50)
        for epoch in range(1, params.train.epochs + 1):
            for batchdata in trainloader:

                action = batchdata["action"]
                cameras = list(batchdata["camera"].values())

                action = action.reshape(-1, *action.shape[-1:])
                for i in range(len(cameras)):
                    # preprocess
                    cameras[i] = cameras[i].reshape(-1, *cameras[i].shape[-3:])
                    # observation = observation.reshape((-1,) + observation.shape[-3:])

                u_t = action
                # x_t = model.cell.q_encoder.given(observation).sample()
                with torch.no_grad():
                    x_t = model.encode(cameras).detach()

                # u_t = scaler_u.pre(u_t)
                # x_t = scaler_x.pre(x_t)

                # print("=================")
                # print("u:", u_t.shape)
                # print("x:", x_t.shape)
                # print("u:", u_t.min().item(), u_t.max().item())
                # print("x:", x_t.min().item(), x_t.max().item())

                # -log P(u_t | x_t)  Eq. (12)
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

            if epoch % params.train.save_per_epoch == 0:
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


if __name__ == "__main__":
    main()
