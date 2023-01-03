"""
Do not import NewtonianVAE series
"""

import builtins
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.ticker import FormatStrFormatter
from torch import nn

import json5
import mypython.plotutil as mpu
import tool.util
from mypython.terminal import Color
from tool import checker, paramsmanager

_weight = "weight*/*"


def select_date(root) -> Optional[Path]:
    if not Path(root).exists():
        Color.print(f"'{root}' doesn't exist.", c=Color.red)
        return None
    if not Path(root).is_dir():
        Color.print(f"'{root}' is not a directory.", c=Color.red)
        return None

    dates = [d for d in Path(root).glob("*") if d.is_dir()]
    dates.sort()

    # _weight の無いディレクトリを除去
    for date in reversed(dates):
        if len(list(date.glob(_weight))) == 0:
            dates.remove(date)

    if len(dates) == 0:
        Color.print(f'"Date and time directory" doesn\'t exist in "{root}" directory.', c=Color.red)
        return None

    for i, date in enumerate(dates, 1):
        l = len(list(date.glob(_weight)))
        _s = (f"{i:2d}", ":", date.name, f"({l:3d})", Path(date, "params_saved.json5"))
        if Preferences.get(date, "running") is None:
            print(*_s)
        else:
            print(*_s, Color.coral + "Running" + Color.reset)

    idx = _get_idx("Select date and time (or exit): ", len(dates))
    if idx is None:
        return None
    else:
        return dates[idx]


def select_weight(path: Path) -> Optional[Path]:
    weight_paths = list(path.glob(_weight))
    weight_paths.sort(key=lambda e: int(e.stem))
    if len(weight_paths) == 0:
        Color.print("Weight doesn't exist.", c=Color.orange)
        return None

    for i, weight_p in enumerate(weight_paths, 1):
        print(f"{i:3d}", ":", weight_p.name)

    idx = _get_idx("Choose weight (or exit): ", len(weight_paths))
    if idx is None:
        return None
    else:
        return weight_paths[idx]


# def delete_useless_saves(root):
#     dates = [d for d in Path(root).glob("*") if d.is_dir()]
#     for date in reversed(dates):
#         if Preferences.get(date, "running") != True and len(list(date.glob(_weight))) == 0:
#             shutil.rmtree(date)


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
    """
    managed_dir (date and time)
    ├── weight_dir
    │   ├── {epoch}.pth (weight_path)
    │   ...
    └── params_saved.json5
    """

    datetime_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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


def load(root: str, model_place):
    manage_dir = tool.util.select_date(root)
    if manage_dir is None:
        sys.exit()
    weight_path = tool.util.select_weight(manage_dir)
    if weight_path is None:
        sys.exit()

    params_path = Path(manage_dir, "params_saved.json5")
    saved_params = paramsmanager.Params(params_path)
    Color.print("params path:", params_path)

    ModelType = getattr(model_place, saved_params.model)
    model: nn.Module = ModelType(**saved_params.model_params)
    model.load_state_dict(torch.load(weight_path))
    return model, manage_dir, weight_path, saved_params


def load_direct(weight_path, model_place):
    weight_path = Path(weight_path)
    manage_dir = weight_path.parent.parent
    params = paramsmanager.Params(Path(manage_dir, "params_saved.json5"))
    ModelType = getattr(model_place, params.model)
    model: nn.Module = ModelType(**params.model_params)
    model.load_state_dict(torch.load(weight_path))
    return model, params


def priority(x1, x2, default=None):
    if x1 is not None:
        return x1
    elif x2 is not None:
        return x2
    else:
        return default


class RecoderBase:
    def append(self, **kwargs):
        for k in self.__dict__.keys():
            self.__dict__[k][-1].append(kwargs[k])

    def add_list(self):
        for k, v in self.__dict__.items():
            v.append([])

    def to_whole_np(self, show_shape=False):
        for k, v in self.__dict__.items():
            self.__dict__[k] = np.stack(v)

        if show_shape:
            max_ = max([len(e) for e in self.__dict__.keys()])
            print("=== Recorded shape ===")
            for k, v in self.__dict__.items():
                print(f"{k}: " + " " * (max_ - len(k)) + f"{v.shape}")
            print("======================")
