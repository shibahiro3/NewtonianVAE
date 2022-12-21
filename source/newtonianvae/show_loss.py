import sys
from pathlib import Path
from typing import List

import classopt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

import mypython.plotutil as mpu
import tool.plot_config
import tool.util
from tool import argset

tool.plot_config.apply()
try:
    import tool._plot_config

    tool._plot_config.apply()
except:
    pass

config = {
    "figure.figsize": (11.39, 3.9),
    "figure.subplot.left": 0.05,
    "figure.subplot.right": 0.98,
    "figure.subplot.bottom": 0.15,
    "figure.subplot.top": 0.85,
    "figure.subplot.wspace": 0.25,
}
plt.rcParams.update(config)


@classopt.classopt(default_long=True, default_short=False)
class Args:
    path_model: str = classopt.config(**argset.descr_path_model, required=True)
    path_result: str = classopt.config(**argset.descr_path_result)
    start_iter: int = classopt.config(metavar="NUM")
    format: List[str] = classopt.config(nargs="*", default=["svg", "pdf"])


args = Args.from_args()  # pylint: disable=E1101


assert args.start_iter > 0


def main():
    # ============================================================
    fig = plt.figure()
    mpu.get_figsize(fig)
    fig.suptitle("Minimize -ELBO = Loss = NLL (= Negative log-likelihood = Recon.) + KL")

    class Ax:
        def __init__(self) -> None:
            gs = GridSpec(nrows=1, ncols=3)
            self.loss = fig.add_subplot(gs[0, 0], title="Loss")
            self.nll = fig.add_subplot(gs[0, 1])
            self.kl = fig.add_subplot(gs[0, 2])

        def clear(self):
            for ax in self.__dict__.values():
                ax.clear()

    axes = Ax()
    # ============================================================

    manage_dir = tool.util.select_date(args.path_model)
    if manage_dir is None:
        return

    loss = np.load(Path(manage_dir, "LOG_Loss.npy"))
    nll = np.load(Path(manage_dir, "LOG_NLL.npy"))
    kl = np.load(Path(manage_dir, "LOG_KL.npy"))

    print("loss len:", len(loss))
    start_idx = args.start_iter

    assert start_idx < len(loss)

    color = "dodgerblue"
    alpha = 0.5
    # span = 50
    span = (len(loss) - start_idx) // 50

    # ============================================================
    data = loss[start_idx:]
    smooth = pd.DataFrame(data).ewm(span=span).mean()
    ax = axes.loss
    ax.set_title("Loss")
    ax.plot(range(start_idx, len(loss)), data, color=color, alpha=alpha)
    ax.plot(range(start_idx, len(loss)), smooth, color=color, lw=2)
    ax.set_xticks([start_idx, len(loss)])
    # sns.lineplot(...)

    # ============================================================
    data = nll[start_idx:]
    smooth = pd.DataFrame(data).ewm(span=span).mean()
    ax = axes.nll
    ax.set_title("NLL")
    ax.set_xlabel("Iterations")
    ax.plot(range(start_idx, len(loss)), data, color=color, alpha=alpha)
    ax.plot(range(start_idx, len(loss)), smooth, color=color, lw=2)
    ax.set_xticks([start_idx, len(loss)])

    # ============================================================
    data = kl[start_idx:]
    smooth = pd.DataFrame(data).ewm(span=span).mean()
    ax = axes.kl
    ax.set_title("KL")
    ax.plot(range(start_idx, len(loss)), data, color=color, alpha=alpha)
    ax.plot(range(start_idx, len(loss)), smooth, color=color, lw=2)
    ax.set_xticks([start_idx, len(loss)])

    # ============================================================
    # fig.tight_layout()

    if args.path_result is not None:
        save_path = Path(args.path_result, f"{manage_dir.stem}_loss.hoge")
        mpu.register_save_path(fig, save_path, args.format)

    plt.show()


if __name__ == "__main__":
    main()
