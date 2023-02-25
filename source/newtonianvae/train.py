import os
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch
from torch import Tensor, nn, optim

import models.core
import mypython.ai.torchprob as tp
import tool.util
from models.core import NewtonianVAEFamily
from mypython.ai.util import (
    SequenceDataLoader,
    print_module_params,
    reproduce,
    show_model_info,
)
from mypython.pyutil import RemainingTime, s2dhms_str
from mypython.terminal import Color, Prompt
from tool import paramsmanager
from tool.util import Preferences, creator, dtype_device
from view.visualhandlerbase import VisualHandlerBase


def train(
    config: str,
    resume: bool,
    vh=VisualHandlerBase(),
):
    torch.set_grad_enabled(True)
    # torch.autograd.detect_anomaly()

    params = paramsmanager.Params(config)

    if params.train.seed is None:
        params.train.seed = np.random.randint(0, 2**16)
    reproduce(params.train.seed)

    dtype, device = dtype_device(
        dtype=params.train.dtype,
        device=params.train.device,
    )

    trainloader = SequenceDataLoader(
        root=Path(params.path.data_dir, "episodes"),
        start=params.train.data_start,
        stop=params.train.data_stop,
        batch_size=params.train.batch_size,
        dtype=dtype,
        device=device,
    )

    model, managed_dir, weight_dir, resume_weight_path = creator(
        root=params.path.saves_dir,
        model_place=models.core,
        model_name=params.model,
        model_params=params.model_params,
        resume=resume,
    )
    model: NewtonianVAEFamily
    model.type(dtype)
    model.to(device)
    model.train()
    # print_module_params(model)
    optimizer = optim.Adam(model.parameters(), params.train.learning_rate)

    params.path.resume_weight = resume_weight_path
    params.pid = os.getpid()
    params.save_train(Path(managed_dir, "params_saved.json5"))

    vh.title = managed_dir.stem
    vh.call_end_init()

    record_losses: Dict[str, list] = {}

    Preferences.put(managed_dir, "running", True)

    def end_process():
        Preferences.remove(managed_dir, "running")

        if len(list(weight_dir.glob("*"))) > 0:
            np.savez(Path(managed_dir, "Losses.npz"), **record_losses)
        else:
            shutil.rmtree(managed_dir)

        print("\nEnd of train")

    show_model_info(model, verbose=False)

    try:
        tp.config.check_value = params.train.check_value  # if False, A little bit faster
        remaining = RemainingTime(max=params.train.epochs * len(trainloader))
        for epoch in range(1, params.train.epochs + 1):

            if params.train.kl_annealing:
                # Paper:
                # In the point mass experiments
                # we found it useful to anneal the KL term in the ELBO,
                # starting with a value of 0.001 and increasing it linearly
                # to 1.0 between epochs 30 and 60.
                if epoch < 30:
                    model.cell.kl_beta = 0.001
                elif 30 <= epoch and epoch <= 60:
                    model.cell.kl_beta = 0.001 + (1 - 0.001) * ((epoch - 30) / (60 - 30))
                else:
                    model.cell.kl_beta = 1

            for batchdata in trainloader:
                batchdata["delta"].unsqueeze_(-1)

                L, losses = model(batchdata)
                L: Tensor
                losses: Dict[str, float]
                for k in losses.keys():
                    if type(losses[k]) == Tensor:
                        losses[k] = losses[k].cpu().item()

                optimizer.zero_grad()
                L.backward()
                # print_module_params(model, True)

                if params.train.grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(
                        model.parameters(), params.train.grad_clip_norm, norm_type=2
                    )

                optimizer.step()

                # === show progress ===

                L = L.cpu().item()

                now_losses = {"Loss": L, **losses}

                for k in now_losses.keys():
                    if k not in record_losses.keys():
                        record_losses[k] = []
                    record_losses[k].append(now_losses[k])

                vh.plot({"Epoch": epoch, **now_losses})
                if not vh.is_running:
                    vh.call_end()
                    time.sleep(0.1)
                    end_process()
                    return

                remaining.update()
                Prompt.print_one_line(
                    f"Epoch: {epoch} | "
                    + " | ".join([f"{k}: {v:.4f}" for k, v in now_losses.items()])
                    + " | "
                    + f"Elapsed: {s2dhms_str(remaining.elapsed)} | "
                    + f"Remaining: {s2dhms_str(remaining.time)} | "
                    + f"ETA: {remaining.eta} "
                )

            if epoch % params.train.save_per_epoch == 0:
                torch.save(model.state_dict(), Path(weight_dir, f"{epoch}.pth"))
                print("saved")

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt")
    except:
        print("=== traceback ===")
        print(traceback.format_exc())

    end_process()
