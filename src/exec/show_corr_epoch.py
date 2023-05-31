import common

common.set_path(__file__)

import argparse
import os
import sys
from argparse import RawTextHelpFormatter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
from matplotlib import ticker

import mypython
import mypython.plotutil as mpu
import tool.util
import view.plot_config
from mypython.ai.train import save_pathname
from mypython.terminal import Color
from mypython.valuewriter import ValueWriter
from tool import paramsmanager


def main():
    # fmt: off
    description = \
"""\
Show correlation by epoch
  Save: Press S key on a window
"""
    parser = argparse.ArgumentParser(allow_abbrev=False, formatter_class=RawTextHelpFormatter, description=description)
    parser.add_argument("-c", "--config", type=str, required=True, **common.config)
    parser.add_argument("--format", type=str, default=common.default_fig_formats, nargs="*", **common.format_file)
    parser.add_argument("--no-window", action='store_true', help="Save instantly without displaying a window")
    args = parser.parse_args()
    # fmt: on

    corr_epoch(**vars(args))


def corr_epoch(
    config: str,
    format: List[str] = ["png", "pdf", "svg"],
    no_window: bool = False,
    # mode: str,
):
    view.plot_config.apply()
    plt.rcParams.update(
        {
            "figure.figsize": (7.59, 4.1),
            "figure.subplot.left": 0.08,
            "figure.subplot.right": 0.98,
            "figure.subplot.bottom": 0.15,
            "figure.subplot.top": 0.85,
            "figure.subplot.wspace": 0.4,
            "lines.linewidth": 1,
        }
    )

    params_path = paramsmanager.Params(config).path
    manage_dir = tool.util.select_date(params_path.saves_dir, no_weight_ok=False, min_epoch=5)
    if manage_dir is None:
        return
    day_time = manage_dir.stem
    datetime.strptime(day_time, "%Y-%m-%d_%H-%M-%S")  # for check (ValueError)

    corr = ValueWriter.load(Path(manage_dir, "corr"))["corr"]

    fig, axes = plt.subplots(1, 1)
    fig.suptitle("Correlation Transition")
    mpu.get_figsize(fig)

    epochs = corr.shape[0]
    dim = corr.shape[-1]

    ax = axes
    ax.hlines(0, 1, epochs, "black")
    ax.hlines([-0.8, 0.8], 1, epochs, "lightgreen", label="$\pm$0.8")
    ax.hlines([-0.9, 0.9], 1, epochs, "red", label="$\pm$0.9")
    for i in range(dim):
        ax.plot(range(1, epochs + 1), corr[:, i], label=f"dim {i+1}")
    # ax.set_xticks(np.linspace(1, epochs, 5, dtype=int))
    ax.set_xlim(1, epochs)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_ylim(-1, 1)
    ax.grid(ls=":")
    # ax.legend()
    mpu.legend_order(ax, list(range(2, 2 + dim)) + [0, 1])

    fig.text(0.5, 0.03, "Epochs", ha="center", va="center", fontsize=14)

    results_dir = params_path.results_dir
    save_path = save_pathname(root=results_dir, day_time=day_time, descr="corr_epoch")
    if no_window:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        for fmt in format:
            p = save_path.with_suffix(f".{fmt}")
            plt.savefig(p)
            Color.print("saved to:", p)
    else:
        mpu.register_save_path(fig, save_path, format)
        plt.show()


if __name__ == "__main__":
    main()
