import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

import mypython.plotutil as mpu
import tool.util
from tool import argset

try:
    import tool._plot_config

    figsize = tool._plot_config.figsize_show_loss
except:
    figsize = None


parser = argparse.ArgumentParser(allow_abbrev=False)
argset.path_model(parser)
argset.start_iter(parser)
argset.format(parser)
argset.path_result(parser, required=False)
_args = parser.parse_args()


class Args:
    path_model = _args.path_model
    path_result = _args.path_result
    start_iter = _args.start_iter
    format = _args.format


args = Args()

assert args.start_iter > 0


def main():
    # ============================================================
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.15, top=0.85)
    mpu.get_figsize(fig)
    fig.suptitle("Minimize -ELBO = Loss = NLL (= Negative log-likelihood = Recon.) + KL")

    class Ax:
        def __init__(self) -> None:
            gs = GridSpec(nrows=1, ncols=3, wspace=0.25)
            self.loss = fig.add_subplot(gs[0, 0])
            self.nll = fig.add_subplot(gs[0, 1])
            self.kl = fig.add_subplot(gs[0, 2])

        def clear(self):
            for ax in self.__dict__.values():
                ax.clear()

    axes = Ax()
    # ============================================================

    d = tool.util.select_date(args.path_model)
    if d is None:
        return

    loss = np.load(Path(d, "LOG_Loss.npy"))
    nll = np.load(Path(d, "LOG_NLL.npy"))
    kl = np.load(Path(d, "LOG_KL.npy"))

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
        save_path = Path(args.path_result, f"{d.stem}_loss.hoge")
        mpu.register_save_path(fig, save_path, args.format)

    plt.show()


if __name__ == "__main__":
    main()
