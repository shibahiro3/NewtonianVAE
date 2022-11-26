import argparse
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import json5
import numpy as np
import torch
from torch import nn, optim

import tool.util
from models.core import get_NewtonianVAE
from mypython.ai.torch_util import print_module_params, reproduce
from mypython.pyutil import s2dhms_str
from tool import argset, checker
from tool.dataloader import DataLoader
from tool.params import Params
from tool.util import Preferences, backup
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

    params = Params(args.cf)
    print("params:")
    print(params)

    vh.title = params.model
    vh.call_end_init()

    if params.train.seed is None:
        seed = np.random.randint(0, 2**16)
    else:
        seed = params.train.seed

    reproduce(seed)

    weight_dir = Path(args.path_save, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), "weight")
    weight_dir.mkdir(parents=True, exist_ok=True)
    backup(args.cf, weight_dir.parent, "params_bk.json5")

    dtype: torch.dtype = getattr(torch, params.train.dtype)
    checker.cuda(params.train.device)
    device = torch.device(params.train.device if torch.cuda.is_available() else "cpu")

    trainloader = DataLoader(
        root=Path(args.path_data, "episodes"),
        start=params.train.data_start,
        stop=params.train.data_stop,
        batch_size=params.train.batch_size,
        dtype=dtype,
        device=device,
    )

    model = get_NewtonianVAE(params.model, **params.raw_[params.model])
    # print_module_params(cell)

    if args.resume:
        print('You chose "resume". Select a model to load.')
        d = tool.util.select_date(args.path_save)
        if d is None:
            return
        weight_p = tool.util.select_weight(d)
        if weight_p is None:
            return
        model.cell.load_state_dict(torch.load(weight_p))

        weight_p_s = str(weight_p)
    else:
        weight_p_s = None

    model.type(dtype)
    model.to(device)
    model.train()

    optimiser = optim.Adam(model.parameters(), params.train.learning_rate)

    Preferences.put(weight_dir.parent, "seed", seed)
    Preferences.put(weight_dir.parent, "resume_from", weight_p_s)
    Preferences.put(weight_dir.parent, "data_from", args.path_data)
    Preferences.put(weight_dir.parent, "data_id", Preferences.get(args.path_data, "id"))

    # LOG_* is a buffer for recording the learning process.
    # It is not used for any calculations for learning.
    # Of course, It has nothing to do with the log function.
    LOG_Loss = []
    LOG_NLL = []
    LOG_KL = []

    def end_process():
        np.save(Path(weight_dir.parent, "LOG_Loss.npy"), LOG_Loss)
        np.save(Path(weight_dir.parent, "LOG_NLL.npy"), LOG_NLL)
        np.save(Path(weight_dir.parent, "LOG_KL.npy"), LOG_KL)

        # ディレクトリを消すのは保存とかの後。
        tool.util.delete_useless_saves(args.path_save)
        print("end of train")

    time_start = time.perf_counter()
    try:
        for epoch in range(1, params.train.epochs + 1):
            for action, observation, delta, position in trainloader:
                E_sum, LOG_E_ll_sum, LOG_E_kl_sum = model(
                    action=action, observation=observation, delta=delta
                )

                L = -E_sum
                optimiser.zero_grad()
                L.backward()
                # print_module_params(cell, True)

                if params.train.grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(
                        model.parameters(), params.train.grad_clip_norm, norm_type=2
                    )

                optimiser.step()

                # === print ===

                L = L.cpu().item()
                LOG_E_ll_sum = -LOG_E_ll_sum.cpu().item()
                LOG_E_kl_sum = LOG_E_kl_sum.cpu().item()

                print(
                    (
                        f"Epoch: {epoch} | "
                        f"Loss: {L:>11.4f} | "
                        f"NLL: {LOG_E_ll_sum:>11.4f} | "
                        f"KL: {LOG_E_kl_sum:>11.4f} | "
                        f"Elapsed: {s2dhms_str(time.perf_counter() - time_start)}"
                    )
                )

                LOG_Loss.append(L)
                LOG_NLL.append(LOG_E_ll_sum)
                LOG_KL.append(LOG_E_kl_sum)

                vh.plot(L, LOG_E_ll_sum, LOG_E_kl_sum, epoch)
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


def save_to_file(path, text):
    p = Path(path)
    with p.open("w") as f:
        f.write(text)
    p.chmod(0o444)


if __name__ == "__main__":
    train()
