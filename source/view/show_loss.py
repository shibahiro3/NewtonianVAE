"""
References:
    https://www.tensorflow.org/tensorboard/dataframe_api?hl=ja
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

import mypython.plotutil as mpu
import tool.util
import view.plot_config
from mypython.valuewriter import ValueWriter
from tool import paramsmanager

view.plot_config.apply()
try:
    import view._plot_config

    view._plot_config.apply()
except:
    pass


def main(
    config: str,
    start_iter: int,
    format: List[str],
    mode: str,
):
    assert start_iter > 0

    plt.rcParams.update(
        {
            "figure.figsize": (11.39, 3.9),
            "figure.subplot.left": 0.05,
            "figure.subplot.right": 0.98,
            "figure.subplot.bottom": 0.15,
            "figure.subplot.top": 0.85,
            "figure.subplot.wspace": 0.4,
        }
    )

    params_path = paramsmanager.Params(config).path
    manage_dir = tool.util.select_date(params_path.saves_dir, no_weight_ok=False)
    if manage_dir is None:
        return

    # losses = np.load(Path(manage_dir, "Losses.npz"), allow_pickle=True)

    if mode == "batch":
        losses = ValueWriter.load(Path(manage_dir, "batch train"))
    elif mode == "epoch":
        losses = ValueWriter.load(Path(manage_dir, "epoch train"))
        losses_valid = ValueWriter.load(Path(manage_dir, "epoch valid"))
    else:
        assert False

    keys = list(losses.keys())
    fig, axes = plt.subplots(1, len(keys))
    mpu.get_figsize(fig)
    fig.suptitle("Loss")

    alpha = 0.5
    start_idx = start_iter - 1

    def plot_axes(losses_, ax: plt.Axes, k, color, label=None):
        steps = len(losses_[k])
        assert start_idx < steps
        span = (steps - start_idx) // 50
        if span < 1:
            span = 1

        data = losses_[k][start_idx:]
        smooth = pd.DataFrame(data).ewm(span=span).mean()
        R = range(start_iter, steps + 1)
        ax.set_title(k)
        ax.plot(R, data, color=color, alpha=alpha)
        ax.plot(R, smooth, color=color, lw=2, label=label)
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xticks(np.linspace(start_iter, steps, 5, dtype=int))  # OK: len(data) < 5
        ax.grid(ls=":")

    if mode == "batch":
        for i, k in enumerate(keys):
            plot_axes(losses, axes[i], k, color="dodgerblue")
        fig.text(0.5, 0.03, "Iterations", ha="center", va="center", fontsize=14)

    elif mode == "epoch":
        for i, k in enumerate(keys):
            plot_axes(losses, axes[i], k, color="dodgerblue", label="train")
            plot_axes(losses_valid, axes[i], k, color="orange", label="valid")
        axes[-1].legend()
        fig.text(0.5, 0.03, "Epochs", ha="center", va="center", fontsize=14)

    # mpu.legend_reduce(fig, loc="lower right")

    if params_path.results_dir is not None:
        save_path = Path(params_path.results_dir, f"{manage_dir.stem}_{mode}_loss")
        mpu.register_save_path(fig, save_path, format)

    plt.show()
