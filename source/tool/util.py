import builtins
import shutil
from pathlib import Path
from typing import Optional

import json5
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import FormatStrFormatter

import mypython.plotutil as mpu
from mypython.terminal import Color

_weight = "weight*/*"


def select_date(model_date_path) -> Optional[Path]:
    if not Path(model_date_path).exists():
        Color.print(f"'{model_date_path}' doesn't exist.", c=Color.red)
        return None
    if not Path(model_date_path).is_dir():
        Color.print(f"'{model_date_path}' is not a directory.", c=Color.red)
        return None

    w_dirs = [d for d in Path(model_date_path).glob("*") if d.is_dir()]
    w_dirs.sort()

    # _weight の無いディレクトリを除去
    for w in reversed(w_dirs):
        if len(list(w.glob(_weight))) == 0:
            w_dirs.remove(w)

    if len(w_dirs) == 0:
        Color.print(
            f'"Date and time directory" doesn\'t exist in "{model_date_path}" directory.',
            c=Color.orange,
        )
        return None

    for i, e in enumerate(w_dirs, 1):
        l = len(list(e.glob(_weight)))
        print(f"{i:2d}", ":", e.name, f"({l:3d})", Path(e, "params_saved.json5"))

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


def save_dict(path, d: dict):
    path = Path(path)
    with open(path, mode="w") as f:
        json5.dump(d, f)
    path.chmod(0o444)


class Label:
    def __init__(self, domain: Optional[str]) -> None:
        self.domain = domain

        self.latent_0 = r"latent element (1)"
        self.latent_1 = r"latent element (2)"

        # TODO:
        self.latent_0_range = None
        self.latent_1_range = None

        if type(domain) == str:
            if domain == "reacher2d":
                self.physical_0 = r"physical angle ($\theta_1$)"
                self.physical_1 = r"physical angle ($\theta_2$)"
                self.physical_0_range = (-np.pi, np.pi)
                self.physical_1_range = (-np.pi / 4, np.pi)

            elif domain == "point_mass":
                self.physical_0 = r"physical position (x)"
                self.physical_1 = r"physical position (y)"
                self.physical_0_range = (-0.3, 0.3)
                self.physical_1_range = (-0.3, 0.3)
            else:
                assert False

        # color_action = mpu.cmap(2, "prism")
        self.color_x = ["#22ff7a", "#e7ad38"]
        self.color_l = ["#16aa4f", "#c59330"]

    def set_axes_L0L1(self, ax: Axes, lmax: Optional[float] = None):
        if self.domain is not None:
            ax.set_xlabel(self.latent_0, color=self.color_l[0])
            if lmax is not None:
                ax.set_xlim(-lmax, lmax)
            # else:
            #     ax.set_xlim()
            ax.xaxis.set_major_formatter(FormatStrFormatter("%3.1f"))

            ax.set_ylabel(self.latent_1, color=self.color_l[1])
            if lmax is not None:
                ax.set_ylim(-lmax, lmax)
            ax.yaxis.set_major_formatter(FormatStrFormatter("%3.1f"))

        mpu.Axis_aspect_2d(ax, 1)

    def set_axes_P0L0(self, ax: Axes, lmax: Optional[float] = None):
        if self.domain is not None:
            ax.set_xlabel(self.physical_0, color=self.color_x[0])
            ax.set_xlim(*self.physical_0_range)

            ax.set_ylabel(self.latent_0, color=self.color_l[0])
            if lmax is not None:
                ax.set_ylim(-lmax, lmax)
            ax.yaxis.set_major_formatter(FormatStrFormatter("%3.1f"))

        mpu.Axis_aspect_2d(ax, 1)

    def set_axes_P1L1(self, ax: Axes, lmax: Optional[float] = None):
        if self.domain is not None:
            ax.set_xlabel(self.physical_1, color=self.color_x[1])
            ax.set_xlim(*self.physical_1_range)

            ax.set_ylabel(self.latent_1, color=self.color_l[1])
            if lmax is not None:
                ax.set_ylim(-lmax, lmax)
            ax.yaxis.set_major_formatter(FormatStrFormatter("%3.1f"))

        mpu.Axis_aspect_2d(ax, 1)

    def set_axes_P0L1(self, ax: Axes, lmax: Optional[float] = None):
        if self.domain is not None:
            ax.set_xlabel(self.physical_0, color=self.color_x[0])
            ax.set_xlim(*self.physical_0_range)

            ax.set_ylabel(self.latent_1, color=self.color_l[1])
            if lmax is not None:
                ax.set_ylim(-lmax, lmax)
            ax.yaxis.set_major_formatter(FormatStrFormatter("%3.1f"))

        mpu.Axis_aspect_2d(ax, 1)

    def set_axes_P1L0(self, ax: Axes, lmax: Optional[float] = None):
        if self.domain is not None:
            ax.set_xlabel(self.physical_1, color=self.color_x[1])
            ax.set_xlim(*self.physical_1_range)

            ax.set_ylabel(self.latent_0, color=self.color_l[0])
            if lmax is not None:
                ax.set_ylim(-lmax, lmax)
            ax.yaxis.set_major_formatter(FormatStrFormatter("%3.1f"))

        mpu.Axis_aspect_2d(ax, 1)
