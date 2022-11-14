import argparse
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils
import torch.utils.data
from torch import nn, optim

import tool.util
from models.core import (
    CollectTimeSeriesData,
    NewtonianVAECell,
    NewtonianVAEDerivationCell,
    get_NewtonianVAECell,
)
from mypython.ai.torch_util import reproduce
from mypython.pyutil import s2dhms_str
from tool import argset, checker
from tool.dataloader import GetBatchData
from tool.params import Params
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
    weight_dir = Path(args.path_save, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), "weight")
    weight_dir.mkdir(parents=True, exist_ok=True)
    bk = Path(weight_dir.parent, Path(args.cf).name)
    shutil.copy(args.cf, bk)
    bk_ = Path(weight_dir.parent, "params_bk.json5")
    bk.rename(bk_)  # 3.7 以前はNoneが返る
    bk_.chmod(0o444)  # read only

    if params.train.seed is None:
        seed = np.random.randint(0, 65535)
        reproduce(seed)

        seed_f = Path(weight_dir.parent, "seed.txt")
        with seed_f.open("w") as f:
            f.write(str(seed))
        seed_f.chmod(0o444)
    else:
        reproduce(params.train.seed)

    # LOG_* は学習経過を見るための専用のバッファ　学習のための計算には一切使用しない
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

    dtype: torch.dtype = getattr(torch, params.train.dtype)

    checker.cuda(params.train.device)
    device = torch.device(params.train.device if torch.cuda.is_available() else "cpu")

    cell = get_NewtonianVAECell(params.model, **params.raw_[params.model])

    if args.resume:
        print('You chose "resume". Select a model to load.')
        d = tool.util.select_date(args.path_save)
        if d is None:
            return
        weight_p = tool.util.select_weight(d)
        if weight_p is None:
            return
        cell.load_state_dict(torch.load(weight_p))

        resume_f = Path(weight_dir.parent, "resume_from.txt")
        with resume_f.open("w") as f:
            f.write(str(weight_p))
        resume_f.chmod(0o444)

    cell.type(dtype)
    cell.to(device)

    collector = CollectTimeSeriesData(
        cell=cell,
        T=params.train.max_time_length,
        dtype=dtype,
    )

    time_start = time.perf_counter()
    try:
        cell.train()
        optimiser = optim.Adam(cell.parameters(), params.train.learning_rate)

        for epoch in range(1, params.train.epochs + 1):
            BatchData = GetBatchData(
                path=args.path_data,
                startN=params.train.data_start,
                stopN=params.train.data_stop,
                BS=params.train.batch_size,
                dtype=dtype,
            )

            for action, observation in BatchData:
                E_sum, LOG_E_ll_sum, LOG_E_kl_sum = collector.run(
                    action=action,
                    observation=observation,
                    device=params.train.device,
                    is_save=False,
                )

                L = -E_sum
                optimiser.zero_grad()
                L.backward()

                if params.train.grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(
                        cell.parameters(), params.train.grad_clip_norm, norm_type=2
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
                torch.save(cell.state_dict(), Path(weight_dir, f"{epoch}.pth"))
                print("saved")

    except KeyboardInterrupt:
        pass

    end_process()


if __name__ == "__main__":
    train()
