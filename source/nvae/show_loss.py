import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

import mypython.plotutil as mpu
import tool.plot_config
import tool.util
from tool import argset

parser = argparse.ArgumentParser(allow_abbrev=False)
argset.path_model(parser)
_args = parser.parse_args()


class Args:
    path_model = _args.path_model


args = Args()

tool.plot_config.apply()


def main():
    d = tool.util.select_date(args.path_model)
    if d is None:
        return

    loss = np.load(Path(d, "LOG_Loss.npy"))
    nll = np.load(Path(d, "LOG_NLL.npy"))
    kl = np.load(Path(d, "LOG_KL.npy"))

    fig = plt.figure()
    gs = GridSpec(nrows=1, ncols=3)

    class Ax:
        def __init__(self) -> None:
            self.loss = fig.add_subplot(gs[0, 0])
            self.nll = fig.add_subplot(gs[0, 1])
            self.kl = fig.add_subplot(gs[0, 2])

        def clear(self):
            for ax in self.__dict__.values():
                ax.clear()

    axes = Ax()

    fig.suptitle("Minimize -ELBO = Loss = NLL (Negative log-likelihood) + KL")

    ax = axes.loss
    ax.plot(loss)
    # ax.set_title("Negative ELBO")
    ax.set_title("Loss")

    ax = axes.nll
    ax.plot(nll)
    # ax.set_title("Negative expected value \nof log-likelihood\n(decoder loss)")
    # ax.set_title("NLL (Negative log-likelihood)")
    ax.set_title("NLL")
    ax.set_xlabel("iterations")

    ax = axes.kl
    ax.plot(kl)
    # ax.set_title("KL divergence")
    ax.set_title("KL")

    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
