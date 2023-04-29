import matplotlib.pyplot as plt

from mypython.pyutil import run_once

try:
    from . import _plot_config  # gitignore (for your configuration)
except:
    _plot_config = None


@run_once
def apply():
    plt.rcParams.update(
        {
            # My _plot_config (platform dependent):
            # "figure.dpi": 100,
            # "savefig.dpi": 200,
            # # "font.family" : "IPAGothic",
            # "font.family": "Times New Roman",
            # "mathtext.fontset": "cm",
            "font.size": 11,
            "figure.titlesize": 17,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            # "figure.labelsize": 10,
            "lines.linestyle": "-",  # ls
            "lines.linewidth": 0.5,  # lw
            # "lines.marker": "o",
            "lines.marker": "None",
            "lines.markersize": 1,  # ms
            "lines.markeredgecolor": "None",  # mec
        }
    )

    if _plot_config is not None:
        _plot_config.apply()

    # print(plt.rcParams)
