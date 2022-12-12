"""
Do not import NewtonianVAE series
"""

import argparse
import builtins
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import json5
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.ticker import FormatStrFormatter
from torch import nn

import mypython.plotutil as mpu
import tool.util
from mypython.terminal import Color
from tool import checker

_weight = "weight*/*"


def select_date(root) -> Optional[Path]:
    if not Path(root).exists():
        Color.print(f"'{root}' doesn't exist.", c=Color.red)
        return None
    if not Path(root).is_dir():
        Color.print(f"'{root}' is not a directory.", c=Color.red)
        return None

    w_dirs = [d for d in Path(root).glob("*") if d.is_dir()]
    w_dirs.sort()

    # _weight の無いディレクトリを除去
    for w in reversed(w_dirs):
        if len(list(w.glob(_weight))) == 0:
            w_dirs.remove(w)

    if len(w_dirs) == 0:
        Color.print(f'"Date and time directory" doesn\'t exist in "{root}" directory.', c=Color.red)
        return None

    for i, e in enumerate(w_dirs, 1):
        l = len(list(e.glob(_weight)))
        _s = (f"{i:2d}", ":", e.name, f"({l:3d})", Path(e, "params_saved.json5"))
        if Preferences.get(root, "running") is None:
            print(*_s)
        else:
            print(*_s, Color.coral + "Running" + Color.reset)

    idx = _get_idx("Select date and time (or exit): ", len(w_dirs))
    if idx is None:
        return None
    else:
        return w_dirs[idx]


def select_weight(path: Path) -> Optional[Path]:
    weight_path = list(path.glob(_weight))
    weight_path.sort(key=lambda e: int(e.stem))
    if len(weight_path) == 0:
        Color.print("Weight doesn't exist.", c=Color.orange)
        return None

    for i, e in enumerate(weight_path, 1):
        print(f"{i:3d}", ":", e.name)

    idx = _get_idx("Choose weight (or exit): ", len(weight_path))
    if idx is None:
        return None
    else:
        return weight_path[idx]


def delete_useless_saves(model_date_path):
    w_dirs = [d for d in Path(model_date_path).glob("*") if d.is_dir()]
    w_dirs.sort()

    # _weight の無いディレクトリを消去
    for w in reversed(w_dirs):
        if len(list(w.glob(_weight))) == 0:
            # w_dirs.remove(w)
            shutil.rmtree(w)


def _get_idx(text, len_list):
    while True:
        idx = builtins.input(Color.green + text + Color.reset)

        if idx == "exit":
            print("Bye!")
            return None
        else:

            try:
                idx = int(idx) - 1
                if 0 <= idx and idx < len_list:
                    return idx
                else:
                    Color.print(f"Please 1 to {len_list}. again.", c=Color.red)
            except ValueError:
                Color.print("Input integer or exit. again.", c=Color.red)


def backup(src_file, dst_dir, rename):
    assert Path(src_file).is_file()
    assert Path(dst_dir).is_dir()

    shutil.copy(src_file, dst_dir)
    bk = Path(dst_dir, Path(src_file).name)
    bk_ = Path(dst_dir, rename)
    bk.rename(bk_)  # 3.7 以前はNoneが返る
    bk_.chmod(0o444)  # read only


def get_data_path(arg_data, trained_time_dir):
    if arg_data is not None:
        data_p = Path(arg_data)
        init_info_p = Path(trained_time_dir, "init_info.json5")
        if init_info_p.exists():
            pass


class Preferences:
    @staticmethod
    def put(dir, name, value):
        dir = Path(dir)
        dir.mkdir(parents=True, exist_ok=True)
        p = Path(dir, f"{name}.json5")
        with open(p, mode="w") as f:
            json5.dump({"value": value}, f)
        p.chmod(0o444)

    @staticmethod
    def get(dir, name):
        ret = None
        p = Path(dir, f"{name}.json5")
        if p.exists():
            with open(p, mode="r") as f:
                ret = json5.load(f).get("value")
        return ret

    @staticmethod
    def remove(dir, name):
        p = Path(dir, f"{name}.json5")
        if p.exists():
            p.unlink()


def save_dict(path, d: dict):
    path = Path(path)
    with open(path, mode="w") as f:
        json5.dump(d, f)
    path.chmod(0o444)


def dtype_device(dtype, device):
    dtype: torch.dtype = getattr(torch, dtype)
    checker.cuda(device)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    return dtype, device


def creator(
    root: str,
    model_place,
    model_name: str,
    model_params: dict,
    resume: bool = False,
):
    datetime_now = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    managed_dir = Path(root, datetime_now)
    weight_dir = Path(managed_dir, "weight")
    weight_dir.mkdir(parents=True, exist_ok=True)

    ModelType = getattr(model_place, model_name)
    model: nn.Module = ModelType(**model_params)

    if resume:
        print('You chose "resume". Select a model to load.')
        resume_manage_dir = tool.util.select_date(root)
        if resume_manage_dir is None:
            sys.exit()
        resume_weight_path = tool.util.select_weight(resume_manage_dir)
        if resume_weight_path is None:
            sys.exit()
        model.load_state_dict(torch.load(resume_weight_path))
    else:
        resume_weight_path = None

    return model, managed_dir, weight_dir, resume_weight_path
