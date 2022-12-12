import argparse
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch import nn, optim

import models.core
import tool.util
from models.core import NewtonianVAE, NewtonianVAEV2, get_NewtonianVAE
from mypython.ai.torch_util import print_module_params, reproduce
from mypython.pyutil import RemainingTime, s2dhms_str
from mypython.terminal import Color, Prompt
from tool import argset, checker, paramsmanager
from tool.dataloader import DataLoader
from tool.util import Preferences, creator, dtype_device
from tool.visualhandlerbase import VisualHandlerBase

parser = argparse.ArgumentParser(allow_abbrev=False)
argset.cf(parser)
argset.path_data(parser)
argset.path_save(parser)
argset.resume(parser)
_args = parser.parse_args()


class Args:
    cf = _args.cf
    path_data = _args.path_data
    path_save = _args.path_save
    resume = _args.resume


args = Args()


def train(vh=VisualHandlerBase()):
    torch.set_grad_enabled(True)

    params = paramsmanager.Params(args.cf)

    if params.train.seed is None:
        params.train.seed = np.random.randint(0, 2**16)
    reproduce(params.train.seed)

    dtype, device = dtype_device(
        dtype=params.train.dtype,
        device=params.train.device,
    )

    trainloader = DataLoader(
        root=Path(args.path_data, "episodes"),
        start=params.train.data_start,
        stop=params.train.data_stop,
        batch_size=params.train.batch_size,
        dtype=dtype,
        device=device,
    )

    model, managed_dir, weight_dir, resume_weight_path = creator(
        root=args.path_save,
        model_place=models.core,
        model_name=params.model,
        model_params=params.raw_[params.model],
        resume=args.resume,
    )
    model: Union[NewtonianVAE, NewtonianVAEV2]
    model.type(dtype)
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), params.train.learning_rate)

    params.external = paramsmanager.TrainExternal(
        data_path=args.path_data,
        data_id=Preferences.get(args.path_data, "id"),
        resume_from=str(resume_weight_path),
    )

    save_param_path = Path(managed_dir, "params_saved.json5")
    params.save(save_param_path)
    Color.print("save params to:", save_param_path)

    vh.title = managed_dir.stem
    vh.call_end_init()

    record_Loss = []
    record_NLL = []
    record_KL = []

    def end_process():
        np.save(Path(managed_dir, "LOG_Loss.npy"), record_Loss)
        np.save(Path(managed_dir, "LOG_NLL.npy"), record_NLL)
        np.save(Path(managed_dir, "LOG_KL.npy"), record_KL)

        tool.util.delete_useless_saves(args.path_save)
        Preferences.remove(managed_dir, "running")
        Color.print("\nEnd of train")

    Preferences.put(managed_dir, "running", True)

    try:
        remaining = RemainingTime(max=params.train.epochs * len(trainloader))
        for epoch in range(1, params.train.epochs + 1):
            for action, observation, delta, position in trainloader:

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

                E, E_ll, E_kl = model(action=action, observation=observation, delta=delta)

                L = -E
                optimizer.zero_grad()
                L.backward()
                # print_module_params(cell, True)

                if params.train.grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(
                        model.parameters(), params.train.grad_clip_norm, norm_type=2
                    )

                optimizer.step()

                # === show progress ===

                L = L.cpu().item()
                E_ll = -E_ll.cpu().item()
                E_kl = E_kl.cpu().item()

                remaining.update()

                Prompt.print_one_line(
                    (
                        f"Epoch: {epoch} | "
                        f"Loss: {L:.4f} | "
                        f"NLL: {E_ll:.4f} | "
                        f"KL: {E_kl:.4f} | "
                        f"Elapsed: {s2dhms_str(remaining.elapsed)} | "
                        # f"Remaining: {s2dhms_str(remaining.time)} "
                        f"ETA: {remaining.eta} "
                    )
                )

                record_Loss.append(L)
                record_NLL.append(E_ll)
                record_KL.append(E_kl)

                vh.plot(L, E_ll, E_kl, epoch)
                if not vh.is_running:
                    vh.call_end()
                    end_process()
                    return

            if epoch % params.train.save_per_epoch == 0:
                torch.save(model.cell.state_dict(), Path(weight_dir, f"{epoch}.pth"))
                print("saved")

    except KeyboardInterrupt:
        pass

    end_process()


if __name__ == "__main__":
    train()
