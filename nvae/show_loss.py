import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

import mypython.plot_config  # noqa: F401
import mypython.plotutil as mpu
import tool.util
from tool import argset

parser = argparse.ArgumentParser(allow_abbrev=False)
argset.path_model(parser)
args = parser.parse_args()


def main():
    d = tool.util.select_date(args.path_model)
    if d is None:
        return

    loss = np.load(Path(d, "LOG_Loss.npy"))
    nll = np.load(Path(d, "LOG_NLL.npy"))
    kl = np.load(Path(d, "LOG_KL.npy"))

    fig = plt.figure()
    gs = GridSpec(nrows=1, ncols=3)
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
    ]

    fig.suptitle("Minimize -ELBO = Loss = NLL (Negative log-likelihood) + KL")

    ax = axes[0]
    ax.plot(loss)
    # ax.set_title("Negative ELBO")
    ax.set_title("Loss")

    ax = axes[1]
    ax.plot(nll)
    # ax.set_title("Negative expected value \nof log-likelihood\n(decoder loss)")
    # ax.set_title("NLL (Negative log-likelihood)")
    ax.set_title("NLL")
    ax.set_xlabel("iterations")

    ax = axes[2]
    ax.plot(kl)
    # ax.set_title("KL divergence")
    ax.set_title("KL")

    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
