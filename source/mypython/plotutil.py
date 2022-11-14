from pathlib import Path
from typing import Iterator, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation


def get_figsize(fig):
    def _callback(event):
        w, h = plt.gcf().get_size_inches()
        print(f"figsize=({w}, {h})")

    fig.canvas.mpl_connect("resize_event", _callback)


def anim_mode(mode: str, fig, anim_func, frames: int, freeze_cnt=0, interval=33, save_path=None):
    """
    Args:
        anim_func:
            args: frame_cnt (int) return None
                frame_cnt: 0, 1, 2, ..., frames-1, 0, 1, 2, ...
                https://qiita.com/t0d4_/items/0f2b41782a6177d35e65

        interval: milliseconds
    """

    assert mode in ("all", "freeze", "anim", "save")

    if mode == "save":
        assert save_path is not None

    print(f"frames: {frames}")

    def null():
        pass

    if mode == "all" or mode == "freeze":
        anim_func(freeze_cnt)
        plt.show()

    elif mode == "anim":
        anim = FuncAnimation(fig, anim_func, frames=frames, init_func=null, interval=interval)
        plt.show()

    elif mode == "save":
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        print("saving...")
        anim = FuncAnimation(fig, anim_func, frames=frames, init_func=null, interval=interval)
        anim.save(save_path)
        print("Done")

    # else:
    #     assert False
