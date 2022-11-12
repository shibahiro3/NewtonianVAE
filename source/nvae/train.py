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
)
from mypython.pyutil import s2dhms_str
from tool import argset
from tool.dataloader import GetBatchData
from tool.params import Params

parser = argparse.ArgumentParser(allow_abbrev=False)
argset.cf(parser)
argset.path_data(parser)
argset.path_save(parser)
_args = parser.parse_args()


class Args:
    cf = _args.cf
    path_data = _args.path_data
    path_save = _args.path_save


args = Args()


def train():
    torch.set_grad_enabled(True)

    params = Params(args.cf)
    print(params)
    weight_dir = Path(args.path_save, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), "weight")

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

    torch_dtype: torch.dtype = getattr(torch, params.train.dtype)
    np_dtype: np.dtype = getattr(np, params.train.dtype)

    if params.train.device == "cuda" and not torch.cuda.is_available():
        print(
            "You have chosen cuda. But your environment does not support cuda, "
            "so this program runs on cpu."
        )
    device = torch.device(params.train.device if torch.cuda.is_available() else "cpu")

    if params.model == "NewtonianVAECell":
        cell = NewtonianVAECell(**params.raw_[params.model])
    elif params.model == "NewtonianVAEDerivationCell":
        cell = NewtonianVAEDerivationCell(**params.raw_[params.model])
    else:
        assert False

    if params.train.resume:
        print('You chose "resume". Select a model to load.')
        d = tool.util.select_date(args.path_save)
        if d is None:
            return
        weight_p = tool.util.select_weight(d)
        if weight_p is None:
            return
        cell.load_state_dict(torch.load(weight_p))

    cell.type(torch_dtype)
    cell.to(device)

    weight_dir.mkdir(parents=True, exist_ok=True)
    bk = Path(weight_dir.parent, Path(args.cf).name)
    shutil.copy(args.cf, bk)
    bk_ = Path(weight_dir.parent, "params_bk.json5")
    bk.rename(bk_)  # 3.7 以前はNoneが返る
    bk_.chmod(0o444)  # read only

    collector = CollectTimeSeriesData(
        cell=cell,
        T=params.train.max_time_length,
        dtype=np_dtype,
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
                dtype=torch_dtype,
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
                        f"epoch: {epoch} | "
                        f"loss: {L:>11.4f} | "
                        f"NLL: {LOG_E_ll_sum:>11.4f} | "
                        f"KL: {LOG_E_kl_sum:>11.4f} | "
                        f"elapsed: {s2dhms_str(time.perf_counter() - time_start)}"
                    )
                )

                LOG_Loss.append(L)
                LOG_NLL.append(LOG_E_ll_sum)
                LOG_KL.append(LOG_E_kl_sum)

            if epoch % params.train.save_per_epoch == 0:
                torch.save(cell.state_dict(), Path(weight_dir, f"{epoch}.pth"))
                print("saved")

    except KeyboardInterrupt:
        pass

    end_process()


if __name__ == "__main__":
    train()
