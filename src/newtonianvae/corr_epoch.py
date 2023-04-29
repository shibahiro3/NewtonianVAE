"""
References:
    https://www.tensorflow.org/tensorboard/dataframe_api?hl=ja
"""


import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import ticker
from matplotlib.gridspec import GridSpec

import mypython.plotutil as mpu
import tool.util
import view.plot_config
from mypython.ai.train import show_loss
from mypython.valuewriter import ValueWriter
from tool import paramsmanager


def main(
    config: str,
    format: List[str] = ["png", "pdf", "svg"],
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
    manage_dir = tool.util.select_date(params_path.saves_dir, no_weight_ok=False)
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
    # ax.xaxis.set_major_locator(ticker.MaxNLocator(5, integer=True))
    ax.set_ylim(-1, 1)
    ax.grid(ls=":")
    # ax.legend()
    mpu.legend_order(ax, list(range(2, 2 + dim)) + [0, 1])

    fig.text(0.5, 0.03, "Epochs", ha="center", va="center", fontsize=14)

    results_dir = params_path.results_dir
    if results_dir is not None:
        save_path = Path(results_dir, day_time, f"{day_time}_corr_epoch")
        mpu.register_save_path(fig, save_path, format)

    plt.show()
