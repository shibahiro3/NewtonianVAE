import matplotlib.pyplot as plt


def apply():
    config = {
        # "font.family" : "IPAGothic",
        "font.family": "Times New Roman",
        "font.size": 11,
        "mathtext.fontset": "cm",
        "savefig.dpi": 300,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linestyle": "-",  # ls
        "lines.linewidth": 0.5,  # lw
        # "lines.marker": "o",
        "lines.marker": "None",
        "lines.markersize": 1,  # ms
        "lines.markeredgecolor": "None",  # mec
    }
    plt.rcParams.update(config)
