from pathlib import Path
from typing import Callable, Iterator, List, Optional, Union

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.backend_bases import KeyEvent
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from mypython.pyutil import add_version
from mypython.terminal import Color


def cmap(N: int, name="rainbow", reverse=False):
    """
    https://matplotlib.org/stable/tutorials/colors/colormaps.html
    """
    color_map = cm.get_cmap(name)
    if reverse:
        return [color_map(i / N) for i in range(N)]
    else:
        return [color_map(1 - i / N) for i in range(N)]


def get_figsize(fig: Figure):
    def _callback(event):
        w, h = plt.gcf().get_size_inches()
        print(f"figsize=({w}, {h})")

    fig.canvas.mpl_connect("resize_event", _callback)


def Axis_aspect_2d(ax: Axes, aspect: float):
    """Put after ax.plot() series
    After:
        ax.set_xlim
        ax.set_ylim
        ax.margins  <- set_x(y)lim と併用不可
    """
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xlen = np.abs(xmax - xmin)
    ylen = np.abs(ymax - ymin)
    ax.set_aspect(aspect * (xlen / ylen))


def cartesian_coordinate(ax: Axes, r: float):
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_aspect(1)
    ax.hlines(0, -r, r, color="black")
    ax.vlines(0, -r, r, color="black")


def register_save_path(
    fig: Figure, path: Union[str, Path], suffixs: Optional[list] = None, version=False
):
    plt.rcParams["keymap.save"] = ""
    path = Path(path)

    if suffixs is None:
        s = path.suffix
        if s == "":
            # raise ValueError("nothing suffix")
            suffixs = ["png"]
        else:
            suffixs = [s]

    def _save(event: KeyEvent):
        if event.key == "s":
            for suffix in suffixs:
                p = path.with_suffix("." + suffix)
                p.parent.mkdir(parents=True, exist_ok=True)
                if version:
                    p = add_version(p)
                event.canvas.figure.savefig(p)
                Color.print("saved to:", p)

    fig.canvas.mpl_connect("key_press_event", _save)


def legend_reduce(fig: Figure, *args, **kwargs):
    """The same label name is regarded as the same expression"""
    labels_handles = {}
    for ax in fig.axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            labels_handles[l] = h

    if len(labels_handles) > 0:
        fig.legend(handles=labels_handles.values(), labels=labels_handles.keys(), *args, **kwargs)


def legend_order(ax: Axes, order: list, *args, **kwargs):
    """
    Ex.
    order=[0, 1, 2]
    """
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], *args, **kwargs)


def anim_mode(
    mode: str,
    fig: Figure,
    anim_func: Callable,
    frames: int,
    freeze_cnt=0,
    interval=33,
    save_path=None,
    repeat=True,
):
    """
    Args:
        anim_func:
            args: frame_cnt (int) return None
                frame_cnt: 0, 1, 2, ..., frames-1, 0, 1, 2, ...

                https://qiita.com/t0d4_/items/0f2b41782a6177d35e65

        interval: milliseconds
    """

    assert mode in ("freeze", "anim", "save")

    if mode == "save":
        assert save_path is not None

    def null():
        pass

    if mode == "freeze":
        anim_func(freeze_cnt)
        plt.show()

    elif mode == "anim":
        anim = FuncAnimation(
            fig, anim_func, frames=frames, init_func=null, interval=interval, repeat=repeat
        )
        plt.show()

    elif mode == "save":
        Color.print("saving to:", save_path)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        anim = FuncAnimation(
            fig, anim_func, frames=frames, init_func=null, interval=interval, repeat=repeat
        )
        anim.save(save_path)
        print("Done")

    # else:
    #     assert False
