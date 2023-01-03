import matplotlib.pyplot as plt


def apply():
    plt.rcParams.update(
        {
            # ====================
            # You can set the following items according to your environment and preferences.
            # I have written them separately in _plot_config.py because I do not want git to track them.
            # "figure.dpi": 100,  # Related to your display size
            # "savefig.dpi": 200,
            # "font.family" : "IPAGothic",
            # "font.family": "Times New Roman",
            # "mathtext.fontset": "cm",
            # ====================
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

    # print(plt.rcParams)
