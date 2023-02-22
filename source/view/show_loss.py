import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

import mypython.plotutil as mpu
import tool.util
import view.plot_config
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
):
    assert start_iter > 0

    plt.rcParams.update(
        {
            "figure.figsize": (11.39, 3.9),
            "figure.subplot.left": 0.05,
            "figure.subplot.right": 0.98,
            "figure.subplot.bottom": 0.15,
            "figure.subplot.top": 0.85,
            "figure.subplot.wspace": 0.25,
        }
    )

    params_path = paramsmanager.Params(config).path
    manage_dir = tool.util.select_date(params_path.saves_dir)
    if manage_dir is None:
        return

    losses = np.load(Path(manage_dir, "Losses.npz"), allow_pickle=True)
    keys = list(losses.keys())
    fig, axes = plt.subplots(1, len(keys))
    mpu.get_figsize(fig)
    fig.suptitle("Minimize -ELBO = Loss = NLL (= Negative log-likelihood = Recon.) + KL")

    color = "dodgerblue"
    alpha = 0.5
    start_idx = start_iter

    for i, k in enumerate(keys):
        steps = len(losses[k])
        assert start_idx < steps
        span = (steps - start_idx) // 50
        # span = 50
        data = losses[k][start_idx:]
        smooth = pd.DataFrame(data).ewm(span=span).mean()
        axes[i].set_title(k)
        # axes[i].set_xlabel("Iterations")
        axes[i].plot(range(start_idx, steps), data, color=color, alpha=alpha)
        axes[i].plot(range(start_idx, steps), smooth, color=color, lw=2)
        axes[i].set_xticks([start_idx, steps])
        # sns.lineplot(...)

    if params_path.results_dir is not None:
        save_path = Path(params_path.results_dir, f"{manage_dir.stem}_loss")
        mpu.register_save_path(fig, save_path, format)

    plt.show()
