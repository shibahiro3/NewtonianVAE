import argparse
import sys
from pathlib import Path
from typing import Dict

import json5
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter

import mypython.plotutil as mpu
import mypython.vision as mv
import tool.util
from mypython.pyutil import Seq, add_version
from mypython.terminal import Color, Prompt
from newtonianvae.load import get_path_data, load
from tool import argset, checker
from tool.dataloader import DataLoader
from view.label import Label

try:
    import tool._plot_config

    figsize = tool._plot_config.figsize_correlation
except:
    figsize = None


parser = argparse.ArgumentParser(allow_abbrev=False)
argset.episodes(parser)
argset.cf_eval(parser)
argset.path_model(parser)
argset.path_data(parser, required=False)
argset.path_result(parser)
argset.fix_xmap_size(parser, required=False)
argset.env_domain(parser)
argset.format(parser)
# argset.fix_xmap_size(parser, required=False)
_args = parser.parse_args()


class Args:
    episodes = _args.episodes
    cf_eval = _args.cf_eval
    path_model = _args.path_model
    path_data = _args.path_data
    path_result = _args.path_result
    fix_xmap_size = _args.fix_xmap_size
    env_domain = _args.env_domain
    format = _args.format


args = Args()

config = {
    "lines.marker": "o",
    "lines.markersize": 1,
    "lines.markeredgecolor": "None",
    "lines.linestyle": "None",
}
plt.rcParams.update(config)


def correlation():
    # ============================================================
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.05, top=0.98)
    mpu.get_figsize(fig)

    class Ax:
        def __init__(self) -> None:
            gs = GridSpec(nrows=2, ncols=3, hspace=0.2, wspace=0.5)
            v1 = Seq()
            # self.physical = fig.add_subplot(gs[0, 0])
            self.latent_map = fig.add_subplot(gs[0:2, 0])

            self.p0l0 = fig.add_subplot(gs[0, 1])
            self.p1l1 = fig.add_subplot(gs[0, 2])

            self.p0l1 = fig.add_subplot(gs[1, 1])
            self.p1l0 = fig.add_subplot(gs[1, 2])

    axes = Ax()
    label = Label(args.env_domain)
    # ============================================================

    torch.set_grad_enabled(False)
    # if args.anim_mode == "save":
    #     checker.large_episodes(args.episodes)

    model, d, weight_path, params, params_eval, dtype, device = load(args.path_model, args.cf_eval)
    data_path = get_path_data(args.path_data, params)

    testloader = DataLoader(
        root=Path(data_path, "episodes"),
        start=params_eval.data_start,
        stop=params_eval.data_stop,
        batch_size=args.episodes,
        dtype=dtype,
        device=device,
    )
    action, observation, delta, position = next(testloader)
    position = position.detach().cpu()

    model.is_save = True

    latent_map = []
    physical = []

    for episode in range(args.episodes):
        model(
            action=action[:, [episode]],
            observation=observation[:, [episode]],
            delta=delta[:, [episode]],
        )
        latent_map_ = model.LOG_x  # (T, dim(u))
        physical_ = position[:, episode]  # (T, dim(u))
        # physical_ = reacher_default2endeffectorpos(physical_)
        # physical_ = reacher_fix_arg_range(physical_)
        corr_p0l0 = np.corrcoef(physical_[:, 0], latent_map_[:, 0])[0, 1]
        corr_p1l1 = np.corrcoef(physical_[:, 1], latent_map_[:, 1])[0, 1]
        print(corr_p0l0, corr_p1l1)
        latent_map.append(latent_map_)
        physical.append(physical_)

    print("=== whole ===")
    latent_map = np.stack(latent_map)
    physical = np.stack(physical)

    BS, T, D = latent_map.shape

    corr_p0l0 = np.corrcoef(physical.reshape((-1, D))[:, 0], latent_map.reshape((-1, D))[:, 0])[
        0, 1
    ]
    corr_p1l1 = np.corrcoef(physical.reshape((-1, D))[:, 1], latent_map.reshape((-1, D))[:, 1])[
        0, 1
    ]
    print(corr_p0l0, corr_p1l1)

    corr_p0l1 = np.corrcoef(physical.reshape((-1, D))[:, 0], latent_map.reshape((-1, D))[:, 1])[
        0, 1
    ]
    corr_p1l0 = np.corrcoef(physical.reshape((-1, D))[:, 1], latent_map.reshape((-1, D))[:, 0])[
        0, 1
    ]
    print(corr_p0l1, corr_p1l0)

    # ============================================================

    color = mpu.cmap(args.episodes, "rainbow")  # per batch color
    # color = ["#377eb880" for _ in range(args.episodes)]

    lmax = args.fix_xmap_size
    # if args.fix_xmap_size is not None:
    #     lmax = args.fix_xmap_size
    # else:
    #     lmax = np.abs([latent_map.min(), latent_map.max()]).max()

    # ============================================================
    x = latent_map[..., 0]
    y = latent_map[..., 1]

    ax = axes.latent_map
    ax.set_title("Latent map")

    for i in range(args.episodes):
        ax.plot(x[i], y[i], color=color[i])
    label.set_axes_L0L1(ax, lmax)

    # ============================================================
    x = physical[..., 0]
    y = latent_map[..., 0]

    ax = axes.p0l0
    ax.set_title(f"Correlation = {corr_p0l0:.4f}")

    for i in range(args.episodes):
        ax.plot(x[i], y[i], color=color[i])
    label.set_axes_P0L0(ax, lmax)

    # ============================================================
    x = physical[..., 1]
    y = latent_map[..., 1]

    ax = axes.p1l1
    ax.set_title(f"Correlation = {corr_p1l1:.4f}")

    for i in range(args.episodes):
        ax.plot(x[i], y[i], color=color[i])
    label.set_axes_P1L1(ax, lmax)

    # ============================================================
    x = physical[..., 0]
    y = latent_map[..., 1]

    ax = axes.p0l1
    ax.set_title(f"Correlation = {corr_p0l1:.4f}")

    for i in range(args.episodes):
        ax.plot(x[i], y[i], color=color[i])
    label.set_axes_P0L1(ax, lmax)

    # ============================================================
    x = physical[..., 1]
    y = latent_map[..., 0]

    ax = axes.p1l0
    ax.set_title(f"Correlation = {corr_p1l0:.4f}")

    for i in range(args.episodes):
        ax.plot(x[i], y[i], color=color[i])
    label.set_axes_P1L0(ax, lmax)

    # ============================================================
    save_path = Path(args.path_result, f"{d.stem}_W{weight_path.stem}_correlation.pdf")
    # save_path = add_version(save_path)
    mpu.register_save_path(fig, save_path, args.format)

    plt.show()


if __name__ == "__main__":
    correlation()
