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
import view.plot_config
from _private import unet
from models.core import NewtonianVAEBase
from mypython import rdict
from mypython.ai import train as mp_train
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
        params.train.seed = np.random.randint(0, 2**16 - 1)
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
    model: NewtonianVAEBase
    model.type(dtype)
    model.to(device)
    model.train()
    # print_module_params(model)
    optimizer = optim.Adam(model.parameters(), params.train.learning_rate)

    params.path.resume_weight = resume_weight_path
    params.pid = os.getpid()
    saved_params_path = Path(managed_dir, "params_saved.json5")
    params.save_train(saved_params_path)

    vh.title = managed_dir.stem
    vh.call_end_init()

    # TODO: enhance exception
    # Preferences.put(managed_dir, "running", True)

    show_model_info(model, verbose=False)

    tp.config.check_value = params.train.check_value  # if False, A little bit faster

    def pre_epoch_fn(epoch: int):
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

    if params.others.get("use_unet", False):
        pre_unet = unet.MobileUNet(out_channels=1).to(device)

        if resume:
            p_ = Path(
                resume_weight_path.parent.parent,
                "unet_with_nvae",
                "weight",
                resume_weight_path.name,
            )
            pre_unet.load_state_dict(torch.load(p_))
        else:
            p_ = Path(
                params.path.saves_dir,
                "unet",
                "weight.pth",
            )
            pre_unet.load_state_dict(torch.load(p_))

        # pre_unet.train()
        pre_unet.eval()

    def pre_batch_fn(epoch: int, batchdata):
        batchdata["delta"].unsqueeze_(-1)

        if params.others.get("use_unet", False):
            with torch.no_grad():
                T, B, C, H, W = batchdata["camera"]["self"].shape
                batchdata["camera"]["self"] = unet.pre(
                    pre_unet, batchdata["camera"]["self"].reshape(-1, C, H, W)
                ).reshape(T, B, C, H, W)

        return batchdata

    def post_batch_fn(epoch: int):
        print(
            "\n"
            # + Prompt.del_line
            + Color.green
            + f"saved params: {saved_params_path} "
            + Color.reset
            + Prompt.cursor_up(2)
        )

    def post_epoch_fn(epoch: int):
        if epoch % params.train.save_per_epoch == 0:
            p_ = Path(
                params.path.saves_dir,
                managed_dir.stem,
                "unet_with_nvae",
                "weight",
                f"{epoch}.pth",
            )
            p_.parent.mkdir(parents=True, exist_ok=True)
            torch.save(pre_unet.state_dict(), p_)
            print("save unet")

    def results_fn(epoch: int):
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
            save_path=add_version(
                Path(params.path.results_dir, managed_dir.stem, f"E{epoch}_correlation.png")
            ),
            show=False,
            **params.raw["correlation"].get("kwargs", {}),
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
                    f"E{epoch}_reconstructed." + str(params.raw["reconstruction"]["format"]),
                )
            ),
            **params.raw["reconstruction"].get("kwargs", {}),
        )

    mp_train.train(
        model=model,
        optimizer=optimizer,
        trainloader=trainloader,
        validloader=validloader,
        epochs=params.train.epochs,
        managed_dir=managed_dir,
        save_per_epoch=params.train.save_per_epoch,
        results_per=results_per,
        grad_clip_norm=params.train.grad_clip_norm,
        vh=vh,
        pre_epoch_fn=pre_epoch_fn,
        pre_batch_fn=pre_batch_fn,
        post_batch_fn=post_batch_fn,
        post_epoch_fn=post_epoch_fn,
        results_fn=results_fn,
    )
