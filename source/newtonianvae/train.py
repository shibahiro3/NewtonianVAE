import os
import shutil
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from torch import Tensor, nn, optim

import models.core
import mypython.ai.torchprob as tp
import tool.util
from models.core import NewtonianVAEFamily
from mypython import rdict
from mypython.ai.util import (
    SequenceDataLoader,
    print_module_params,
    reproduce,
    show_model_info,
)
from mypython.pyutil import MovingAverageTime, RemainingTime, add_version, s2dhms_str
from mypython.terminal import Color, Prompt
from mypython.valuewriter import ValueWriter
from tool import paramsmanager
from tool.util import Preferences, creator, dtype_device
from view.visualhandlerbase import VisualHandlerBase

from .correlation import correlation_
from .reconstruct import reconstruction_


def train(
    config: str,
    resume: bool,
    results_per: Optional[int] = None,
    vh=VisualHandlerBase(),
):
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

    # Hold-out Validation
    validloader = SequenceDataLoader(
        root=Path(params.path.data_dir, "episodes"),
        start=params.valid.data_start,
        stop=params.valid.data_stop,
        batch_size=params.valid.batch_size,
        dtype=dtype,
        device=device,
    )

    # rdict.show(trainloader.sample_batch(), "train batch")

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

    # TODO: enhance exception
    # Preferences.put(managed_dir, "running", True)

    batch_train_writer = ValueWriter(Path(managed_dir, "batch train"))
    epoch_train_writer = ValueWriter(Path(managed_dir, "epoch train"))
    epoch_valid_writer = ValueWriter(Path(managed_dir, "epoch valid"))

    show_model_info(model, verbose=False)

    try:
        tp.config.check_value = params.train.check_value  # if False, A little bit faster

        time_start_learning = time.perf_counter()
        time_first_training = MovingAverageTime(5)
        time_epoch_training = MovingAverageTime(1)
        for epoch in range(1, params.train.epochs + 1):
            torch.set_grad_enabled(True)
            # torch.autograd.detect_anomaly()

            time_start_epoch = time.perf_counter()

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

            epoch_losses_all = {"train": {}, "valid": {}}
            for phase in ["train", "valid"]:
                epoch_losses = {}

                if phase == "train":
                    dataloader = trainloader
                    model.train()
                else:
                    dataloader = validloader
                    model.eval()

                for batchdata in dataloader:
                    batchdata["delta"].unsqueeze_(-1)

                    L, losses = model(batchdata)
                    L: Tensor
                    losses: Dict[str, float]
                    for k in losses.keys():
                        if type(losses[k]) == Tensor:
                            losses[k] = losses[k].cpu().item()

                    if phase == "train":
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

                    for k in now_losses.keys():
                        if not k in epoch_losses:
                            epoch_losses[k] = now_losses[k]
                        else:
                            epoch_losses[k] += now_losses[k]

                    if phase == "train":
                        for k, v in now_losses.items():
                            batch_train_writer.write(k, v)

                    vh.plot(dict(mode="now", ephoch=epoch, losses=now_losses, phase=phase))

                    if not vh.is_running:
                        vh.call_end()
                        time.sleep(0.1)
                        # end_process()
                        return

                    bt_msg = (
                        f"{phase} | Epoch: {epoch} | "
                        + " | ".join([f"{k}: {v:.4f}" for k, v in now_losses.items()])
                        + " | "
                        + f"Elapsed: {s2dhms_str(time.perf_counter() - time_start_learning)} "
                    )

                    if epoch == 1:
                        if phase == "train":
                            remaining = (
                                time_first_training.update()
                                * params.train.epochs
                                * len(trainloader)
                            )

                        bt_msg += (
                            f"| Remaining: {s2dhms_str(remaining)}+ | "
                            + f"ETA: "
                            + (datetime.now() + timedelta(seconds=remaining)).strftime(
                                "%m/%d %H:%M+ "
                            )
                        )

                    else:
                        remaining = time_epoch_training.get() * (params.train.epochs - epoch + 1)
                        bt_msg += (
                            f"| Remaining: {s2dhms_str(remaining)}+ | "
                            + f"ETA: "
                            + (datetime.now() + timedelta(seconds=remaining)).strftime(
                                "%m/%d %H:%M+ "
                            )
                        )

                    Prompt.print_one_line(bt_msg)

                for k, v in epoch_losses.items():
                    epoch_losses[k] = v / len(dataloader)

                ep_msg = f"{phase} | Epoch: {epoch} | " + " | ".join(
                    [f"{k}: {v :.4f}" for k, v in epoch_losses.items()]
                )
                ep_msg = Prompt.del_line + ep_msg

                if phase == "train":
                    if epoch % params.train.save_per_epoch == 0:
                        torch.save(model.state_dict(), Path(weight_dir, f"{epoch}.pth"))
                        ep_msg += " saved"
                else:
                    ep_msg = (
                        Color.coral
                        + ep_msg
                        + Color.reset
                        + f" | Duration: {s2dhms_str(time.perf_counter() - time_start_epoch)}"
                    )

                epoch_losses_all[phase] = epoch_losses

                vh.plot(dict(mode="average", epoch=epoch, losses=epoch_losses, phase=phase))
                print(ep_msg)

            for k, v in epoch_losses_all["train"].items():
                epoch_train_writer.write(k, v)

            for k, v in epoch_losses_all["valid"].items():
                epoch_valid_writer.write(k, v)

            vh.plot(dict(mode="all", epoch=epoch, losses=epoch_losses_all))

            time_epoch_training.update()

            if (results_per is not None) and (epoch % results_per == 0):
                correlation_(
                    model=model,
                    batchdata=SequenceDataLoader(
                        root=Path(params.path.data_dir, "episodes"),
                        start=params.test.data_start,
                        stop=params.test.data_stop,
                        batch_size=params.raw["correlation"]["episodes"],
                        dtype=dtype,
                        device=device,
                        shuffle=False,
                    ).sample_batch(),
                    all=params.raw["correlation"]["all"],
                    save_path=add_version(
                        Path(params.path.results_dir, managed_dir.stem, f"E{epoch}_correlation.png")
                    ),
                    show=False,
                )

                reconstruction_(
                    model=model,
                    batchdata=SequenceDataLoader(
                        root=Path(params.path.data_dir, "episodes"),
                        start=params.test.data_start,
                        stop=params.test.data_stop,
                        batch_size=params.raw["reconstruction"]["episodes"],
                        dtype=dtype,
                        device=device,
                        shuffle=False,
                    ).sample_batch(),
                    save_path=add_version(
                        Path(
                            params.path.results_dir,
                            managed_dir.stem,
                            f"E{epoch}_reconstructed."
                            + str(params.raw["reconstruction"]["format"]),
                        )
                    ),
                )

        print("Total Duration:", s2dhms_str(time.perf_counter() - time_start_learning))

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt")
    except:
        print("=== traceback ===")
        print(traceback.format_exc())
