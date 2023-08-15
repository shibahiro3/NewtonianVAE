"""
Do not import NewtonianVAE series
"""

import builtins
import dataclasses
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.ticker import FormatStrFormatter
from natsort import natsorted
from torch import nn

import json5
import mypython.plotutil as mpu
import mypython.pyutil as pyu
import tool.preprocess
import tool.util
from mypython import rdict
from mypython.terminal import Color
from mypython.valuewriter import ValueWriter
from tool import checker, paramsmanager, prepost
from unet_mask.seg_data import mask_unet

_weight = "weight*/*"


def select_date(root, no_weight_ok=False, min_epoch=1) -> Optional[Path]:
    """
    root
        date_time_1
        date_time_2
        ...
        date_time_n

    See: from mypython.ai.train import train
    """

    if not Path(root).exists():
        Color.print(f"'{root}' doesn't exist.", c=Color.red)
        return None
    if not Path(root).is_dir():
        Color.print(f"'{root}' is not a directory.", c=Color.red)
        return None

    dates = [d for d in Path(root).glob("*") if d.is_dir()]
    dates = natsorted(dates)

    if not no_weight_ok:
        # _weight の無いディレクトリをパスリストから除去
        for date in reversed(dates):
            if len(list(date.glob(_weight))) == 0:
                dates.remove(date)

    date_new = []
    cnt = 0
    for date in dates:
        weights = len(list(date.glob(_weight)))
        epochs = len(ValueWriter.load(Path(date, "epoch train"))["Loss"])

        if epochs < min_epoch:
            continue
        else:
            date_new.append(date)
            cnt += 1

        s = (
            f"{cnt:2d} : {date.name} - weights: {weights:2d}, epochs: {epochs:4d}, config_bk: "
            + str(Path(date, "params_saved.json5"))
        )
        if cnt % 2:
            Color.print(s, c=Color.bg_rgb(30, 70, 70))
        else:
            print(s)

    if len(date_new) == 0:
        Color.print(f'"Date and time directory" doesn\'t exist in "{root}" directory.', c=Color.red)
        return None

    idx = _get_idx("Select date and time (or exit): ", len(date_new))
    if idx is None:
        return None
    else:
        return date_new[idx]


def select_weight(root: Path) -> Optional[Path]:
    """
    root
        {number}.{ext} or {number}_*.{ext}
    """

    weight_paths = list(root.glob(_weight))
    weight_paths = natsorted(weight_paths)
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


def _get_idx(text, n_max):
    while True:
        idx = builtins.input(Color.green + text + Color.reset)

        if idx == "exit":
            print("Bye!")
            return None
        else:
            try:
                idx = int(idx) - 1
                if 0 <= idx and idx < n_max:
                    return idx
                else:
                    Color.print(f"Please 1 to {n_max}. again.", c=Color.red)
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
    ├── params_saved.json5
    ├── weight_dir
    │   ├── {epoch}.pth (weight_path)
    │   ...
    │
    ├── loss root
    │   ├── ...
    │

    resume_weight_path
      = old_managed_dir/weight/{epoch}.pth
    """

    datetime_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    managed_dir = Path(root, datetime_now)
    weight_dir = Path(managed_dir, "weight")
    weight_dir.mkdir(parents=True, exist_ok=True)

    model: nn.Module = getattr(model_place, model_name)(**model_params)

    if resume:
        print('You chose "resume". Select a model to load.')
        resume_manage_dir = tool.util.select_date(root, min_epoch=5)
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
    manage_dir = tool.util.select_date(root, min_epoch=5)
    if manage_dir is None:
        sys.exit()
    weight_path = tool.util.select_weight(manage_dir)
    if weight_path is None:
        sys.exit()

    params_path = Path(manage_dir, "params_saved.json5")
    saved_params = paramsmanager.Params(params_path)
    Color.print("params path:", params_path)

    model: nn.Module = getattr(model_place, saved_params.model)(**saved_params.model_params)
    model.load_state_dict(torch.load(weight_path))
    return model, manage_dir, weight_path, saved_params


def load_direct(weight_path, model_place):
    weight_path = Path(weight_path)
    manage_dir = weight_path.parent.parent
    params = paramsmanager.Params(Path(manage_dir, "params_saved.json5"))
    model: nn.Module = getattr(model_place, params.model)(**params.model_params)
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


def create_prepostprocess(params: paramsmanager.Params, device):
    preprocess_list = []

    pp_simple = getattr(tool.preprocess, params.others.get("preprocess", ""), lambda x: x)
    preprocess_list.append(pp_simple)

    pp_image = prepost.HandyForImage(**rdict.get(params.others, ["HandyForImage"], {}))

    def pp_image_wrap(batchdata):
        rdict.apply_(batchdata["camera"], pp_image.pre)
        return batchdata

    preprocess_list.append(pp_image_wrap)

    pp_unet = None
    unet_mask_path = params.others.get("unet_mask", None)
    if unet_mask_path is not None:
        masker, preprocess_unet = mask_unet(unet_mask_path)
        masker.to(device)

        def pp_unet(batchdata):
            return preprocess_unet(batchdata)

        preprocess_list.append(pp_unet)

    class _postrocesses:
        def __init__(self) -> None:
            self.image = pp_image.post

    def preprocess(batchdata):
        for p_ in preprocess_list:
            batchdata = p_(batchdata)
        return batchdata

    return preprocess, _postrocesses()
