import os
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from torch import Tensor, nn, optim

import models.core
import mypython.ai.torchprob as tp
import tool.preprocess
import tool.util
from models import cell, core, parts
from models.core import NewtonianVAEBase
from mypython import rdict
from mypython.ai import train as mp_train
from mypython.ai.util import (
    SequenceDataLoader,
    print_module_params,
    reproduce,
    show_model_info,
)
from mypython.keyboard import Key, KeyInput
from mypython.terminal import Color, Prompt
from mypython.valuewriter import ValueWriter
from tool import paramsmanager
from tool.util import creator, dtype_device
from view.visualhandlerbase import VisualHandlerBase

from .correlation import correlation_, correlation_cal, print_corr
from .reconstruct import reconstruction_


def train(
    config: str,
    resume: bool,
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
        patterns=params.train.path,
        batch_size=params.train.batch_size,
        max_time=params.train.max_time_length,
        dtype=dtype,
        device=device,
        preprocess=getattr(tool.preprocess, params.others.get("preprocess", ""), None),
    )
    trainloader.sample_batch(verbose=True, print_name="sample train batch")

    validloader = SequenceDataLoader(
        patterns=params.valid.path,
        batch_size=params.valid.batch_size,
        dtype=dtype,
        device=device,
        preprocess=getattr(tool.preprocess, params.others.get("preprocess", ""), None),
    )
    validloader.sample_batch(verbose=True, print_name="sample valid batch")

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

    show_model_info(model, verbose=False)

    tp.config.check_value = params.train.check_value  # Faster if False

    keyinput = KeyInput()

    pretrain: dict = params.others.get("pretrain", None)
    if (pretrain is not None) and pretrain["use"]:
        if type(model) == core.MNVAE:
            pre_ee1_state: Dict[str, Tensor] = torch.load(Path("_pre_ee1", "weight", "weight.pth"))
            pre_ee2_state: Dict[str, Tensor] = torch.load(Path("_pre_ee2", "weight", "weight.pth"))

            encs = [
                {k[4:]: v for k, v in pre_ee1_state.items() if k.startswith("enc.")},
                {k[4:]: v for k, v in pre_ee2_state.items() if k.startswith("enc.")},
            ]
            decs = [
                {k[4:]: v for k, v in pre_ee1_state.items() if k.startswith("dec.")},
                {k[4:]: v for k, v in pre_ee2_state.items() if k.startswith("dec.")},
            ]

            if pretrain["encoder"]["use"]:
                for i, m in enumerate(model.cell.q_encoder.encoders):
                    m: parts.ResnetCWrap
                    m.core.load_state_dict(encs[i])
                    m.core.requires_grad_(not pretrain["encoder"]["freeze"])
                    print("OK: ResnetCWrap", i)

            if pretrain["decoder"]["use"]:
                for i, m in enumerate(model.cell.p_decoder.decoders):
                    m: parts.DecoderCWrap
                    m.core.load_state_dict(decs[i])
                    m.core.requires_grad_(not pretrain["decoder"]["freeze"])
                    print("OK: DecoderCWrap", i)

    corr_writer = ValueWriter(Path(managed_dir, "corr"))

    def pre_epoch_fn(epoch: int):

        decoder_free = params.others.get("decoder_free", None)
        if decoder_free is not None:
            if epoch > decoder_free:
                model.cell.set_decoder_free(True)

    # if params.others.get("use_unet", False):
    #     pre_unet = unet.MobileUNet(out_channels=1).to(device)

    #     if resume:
    #         p_ = Path(
    #             resume_weight_path.parent.parent,
    #             "unet_with_nvae",
    #             "weight",
    #             resume_weight_path.name,
    #         )
    #         pre_unet.load_state_dict(torch.load(p_))
    #     else:
    #         p_ = Path(
    #             params.path.saves_dir,
    #             "unet",
    #             "weight.pth",
    #         )
    #         pre_unet.load_state_dict(torch.load(p_))

    #     # pre_unet.train()
    #     pre_unet.eval()

    def pre_batch_fn(epoch: float, batchdata):
        # rdict.show(batchdata, "batchdata")

        beta_schedule = params.others.get("beta_schedule", None)
        if beta_schedule is not None:
            ver = list(beta_schedule.keys())[0]
            items = beta_schedule[ver]
            if ver == "linear1":
                a, b, av, bv = items
                model.cell.kl_beta = partially_linear(epoch, a, b, av, bv)
                # model.cell.kl_beta = partially_linear(epoch, 30, 60, 0.001, 1)

            # Color.print("kl_beta:", model.cell.kl_beta)

        alpha_schedule = params.others.get("alpha_schedule", None)
        if alpha_schedule is not None:
            ver = list(alpha_schedule.keys())[0]
            items = alpha_schedule[ver]
            if ver == "linear1":
                a, b, av, bv = items
                model.cell.alpha = partially_linear(epoch, a, b, av, bv)

            print("alpha:", model.cell.alpha)

        if keyinput.get(block=False) == "p":
            print(Prompt.cursor_down(2))
            print("[Pause]")
            print("Press 'r' to resume")

            rdict.to_torch(batchdata, device=torch.device("cpu"))
            model.cpu()
            torch.cuda.empty_cache()

            keyinput.wait_until("r")
            model.to(device=device)
            print("[Resume]")

        batchdata["delta"].unsqueeze_(-1)

        beta_schedule = params.others.get("beta_schedule", None)
        if beta_schedule is not None:
            ver = list(beta_schedule.keys())[0]
            items = beta_schedule[ver]
            if ver == "linear1":
                a, b, av, bv = items
                model.cell.kl_beta = partially_linear(epoch, a, b, av, bv)
            elif ver == "cyclical_linear":
                span, min, max = items
                model.cell.kl_beta = cyclical_linear(epoch, span, min, max)

            # print("beta:", model.cell.kl_beta)

        # if params.others.get("use_unet", False):
        #     with torch.no_grad():
        #         T, B, C, H, W = batchdata["camera"]["self"].shape
        #         batchdata["camera"]["self"] = unet.pre(
        #             pre_unet, batchdata["camera"]["self"].reshape(-1, C, H, W)
        #         ).reshape(T, B, C, H, W)

        rdict.to_torch(batchdata, device=device)
        return batchdata

    def post_batch_fn(epoch: float, status: dict) -> None:
        print(
            "\n"
            # + Prompt.del_line
            + Color.green
            + f"saved params: {saved_params_path} "
            + Color.reset
            + Prompt.cursor_up(2)
        )

    # mutable
    best = {"corr": np.zeros(model.dim_x), "weight_name": "1_best.pth"}

    def post_epoch_fn(epoch: int, status: dict):
        vh.plot(status)

        # if params.others.get("use_unet", False):
        #     if epoch % params.train.save_per_epoch == 0:
        #         p_ = Path(
        #             params.path.saves_dir,
        #             managed_dir.stem,
        #             "unet_with_nvae",
        #             "weight",
        #             f"{epoch}.pth",
        #         )
        #         p_.parent.mkdir(parents=True, exist_ok=True)
        #         torch.save(pre_unet.state_dict(), p_)
        #         print("save unet")

        bd = validloader.sample_batch(batch_size="all")
        bd["delta"].unsqueeze_(-1)
        corr, _, _ = correlation_cal(model=model, batchdata=bd, all=False)
        corr_writer.write("corr", corr)
        print_corr(corr)
        if (np.abs(corr).sum() > np.abs(best["corr"]).sum()).all():
            best["corr"] = corr
            p = Path(managed_dir, "weight", best["weight_name"])
            if p.exists():
                os.remove(p)  # remove before
            best["weight_name"] = f"{epoch}_best_corr.pth"
            p = Path(managed_dir, "weight", best["weight_name"])
            torch.save(model.state_dict(), p)
            Color.print("Best:", p, c=Color.red)

        # keyinput.reset_cache()

    mp_train.train(
        model=model,
        optimizer=optimizer,
        trainloader=trainloader,
        validloader=validloader,
        epochs=params.train.epochs,
        managed_dir=managed_dir,
        save_per_epoch=params.train.save_per_epoch,
        grad_clip_norm=params.train.grad_clip_norm,
        pre_epoch_fn=pre_epoch_fn,
        pre_batch_fn=pre_batch_fn,
        post_batch_fn=post_batch_fn,
        post_epoch_fn=post_epoch_fn,
    )


def partially_linear(x, a, b, av, bv):
    """
    _/‾ or ‾\_
    """
    if x < a:
        return av
    elif a <= x and x <= b:
        return av + (bv - av) * ((x - a) / (b - a))
    else:
        return bv


def cyclical_linear(x, span, min, max):
    """https://arxiv.org/abs/1903.10145
    /‾|/‾|/‾| ...
    """
    middle = span / 2
    x = x - (x // span) * span
    return partially_linear(x, 0, middle, min, max)
