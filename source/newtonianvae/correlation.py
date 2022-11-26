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
from mypython.plotutil import Axis_aspect_2d, cmap
from mypython.terminal import Color, Prompt
from newtonianvae.load import get_path_data, load
from tool import argset, checker
from tool.dataloader import DataLoader

try:
    import tool._plot_config

    figsize = tool._plot_config.figsize_correlation
except:
    figsize = None


parser = argparse.ArgumentParser(allow_abbrev=False)
argset.episodes(parser)
argset.anim_mode(parser)
argset.cf_eval(parser)
argset.path_model(parser)
argset.path_data(parser, required=False)
argset.path_result(parser)
# argset.fix_xmap_size(parser, required=False)
_args = parser.parse_args()


class Args:
    episodes = _args.episodes
    anim_mode = _args.anim_mode
    cf_eval = _args.cf_eval
    path_model = _args.path_model
    path_data = _args.path_data
    path_result = _args.path_result
    # fix_xmap_size = _args.fix_xmap_size


args = Args()


def correlation():
    if args.anim_mode == "save":
        checker.large_episodes(args.episodes)

    torch.set_grad_enabled(False)

    model, d, weight_p, params, params_eval, dtype, device = load(args.path_model, args.cf_eval)
    data_path = get_path_data(args.path_data, d)

    Path(args.path_result).mkdir(parents=True, exist_ok=True)

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
        corr_x = np.corrcoef(physical_[:, 0], latent_map_[:, 0])[0, 1]
        corr_y = np.corrcoef(physical_[:, 1], latent_map_[:, 1])[0, 1]
        print(corr_x, corr_y)
        latent_map.append(latent_map_)
        physical.append(physical_)

    print("=== whole ===")
    latent_map = np.stack(latent_map)
    physical = np.stack(physical)

    BS, T, D = latent_map.shape

    corr_x = np.corrcoef(physical.reshape((-1, D))[:, 0], latent_map.reshape((-1, D))[:, 0])[0, 1]
    corr_y = np.corrcoef(physical.reshape((-1, D))[:, 1], latent_map.reshape((-1, D))[:, 1])[0, 1]
    print(corr_x, corr_y)

    # =====================================================

    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.07, top=0.95)
    mpu.get_figsize(fig)

    class Ax:
        def __init__(self) -> None:
            gs = GridSpec(nrows=4, ncols=1, hspace=0.8)
            self.physical = fig.add_subplot(gs[0, 0])
            self.latent_map = fig.add_subplot(gs[1, 0])
            self.x = fig.add_subplot(gs[2, 0])
            self.y = fig.add_subplot(gs[3, 0])

    axes = Ax()

    batch_color = cmap(args.episodes, "rainbow")

    s = 0.1

    l_latent_0 = r"latent element (0)"
    l_latent_1 = r"latent element (1)"

    # endeffector
    # l_physical_0 = r"physical position (x)"
    # l_physical_1 = r"physical position (y)"

    l_physical_0 = r"physical angle ($\theta_1$)"
    l_physical_1 = r"physical angle ($\theta_2$)"

    ax = axes.latent_map
    x = latent_map[..., 0]
    y = latent_map[..., 1]
    ax.set_title("Latent map")
    ax.set_xlabel(l_latent_0)
    ax.set_ylabel(l_latent_1)
    # ax.scatter(x.flatten(), y.flatten(), s=s)
    for i in range(args.episodes):
        ax.scatter(x[i], y[i], s=s, color=batch_color[i])
    Axis_aspect_2d(ax, 0.8)

    ax = axes.physical
    x = physical[..., 0]
    y = physical[..., 1]
    ax.set_title("Physical")
    ax.set_xlabel(l_physical_0)
    ax.set_ylabel(l_physical_1)
    # ax.scatter(x.flatten(), y.flatten(), s=s)
    for i in range(args.episodes):
        ax.scatter(x[i], y[i], s=s, color=batch_color[i])
    Axis_aspect_2d(ax, 0.8)

    ax = axes.x
    x = physical[..., 0]
    y = latent_map[..., 0]
    ax.set_title(f"Correlation = {corr_x:.4f}")
    ax.set_xlabel(l_physical_0)
    ax.set_ylabel(l_latent_0)
    # ax.scatter(x.flatten(), y.flatten(), s=s)
    for i in range(args.episodes):
        ax.scatter(x[i], y[i], s=s, color=batch_color[i])
    Axis_aspect_2d(ax, 0.8)

    ax = axes.y
    x = physical[..., 1]
    y = latent_map[..., 1]
    ax.set_title(f"Correlation = {corr_y:.4f}")
    ax.set_xlabel(l_physical_1)
    ax.set_ylabel(l_latent_1)
    # ax.scatter(x.flatten(), y.flatten(), s=s)
    for i in range(args.episodes):
        ax.scatter(x[i], y[i], s=s, color=batch_color[i])
    Axis_aspect_2d(ax, 0.8)

    plt.show()


if __name__ == "__main__":
    correlation()
