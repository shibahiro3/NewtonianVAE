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
from models.core import (
    NewtonianVAECell,
    NewtonianVAEDerivationCell,
    as_save,
    get_NewtonianVAECell,
)
from mypython.plotutil import Axis_aspect_2d, cmap
from mypython.terminal import Prompt
from newtonianvae.load import load
from simulation.env import obs2img
from tool import argset, checker
from tool.dataloader import DataLoader
from tool.params import Params, ParamsEval

try:
    import tool._plot_config

    figsize = tool._plot_config.figsize_reconstruct
except:
    figsize = None


parser = argparse.ArgumentParser(allow_abbrev=False)
argset.episodes(parser)
argset.anim_mode(parser)
argset.cf_eval(parser)
argset.path_model(parser)
argset.path_data(parser)
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
    checker.is_same_data(args.path_data, weight_p.parent.parent)

    Path(args.path_result).mkdir(parents=True, exist_ok=True)

    all_steps = params.train.max_time_length * args.episodes

    testloader = DataLoader(
        root=Path(args.path_data, "episodes"),
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

    for i in range(args.episodes):
        model(action=action[:, [i]], observation=observation[:, [i]], delta=delta[:, [i]])
        latent_map_ = model.LOG_x
        physical_ = position[:, i]
        corr_x = np.corrcoef(physical_[:, 0], latent_map_[:, 0])[0, 1]
        corr_y = np.corrcoef(physical_[:, 1], latent_map_[:, 1])[0, 1]
        print(corr_x, corr_y)
        latent_map.append(latent_map_)
        physical.append(physical_)

    print("=== whole ===")
    latent_map = np.concatenate(latent_map)
    physical = np.concatenate(physical)
    print(latent_map.shape)
    print(physical.shape)
    corr_x = np.corrcoef(physical[:, 0], latent_map[:, 0])[0, 1]
    corr_y = np.corrcoef(physical[:, 1], latent_map[:, 1])[0, 1]
    print(corr_x, corr_y)

    fig = plt.figure(figsize=(4.76, 7.56))
    mpu.get_figsize(fig)

    class Ax:
        def __init__(self) -> None:
            gs = GridSpec(nrows=4, ncols=1, hspace=0.7)
            self.physical = fig.add_subplot(gs[0, 0])
            self.latent_map = fig.add_subplot(gs[1, 0])
            self.x = fig.add_subplot(gs[2, 0])
            self.y = fig.add_subplot(gs[3, 0])

    axes = Ax()

    s = 0.1
    labelsize = 8

    l_latent_0 = r"latent element (0)"
    l_latent_1 = r"latent element (1)"

    # endeffector
    # l_physical_0 = r"physical position (x)"
    # l_physical_1 = r"physical position (y)"

    l_physical_0 = r"physical angle ($\theta_1$)"
    l_physical_1 = r"physical angle ($\theta_2$)"

    ax = axes.latent_map
    x = latent_map[:, 0]
    y = latent_map[:, 1]
    ax.set_title("Latent map")
    ax.set_xlabel(l_latent_0, fontsize=labelsize)
    ax.set_ylabel(l_latent_1, fontsize=labelsize)
    ax.scatter(x, y, s=s)
    Axis_aspect_2d(ax, 0.8, x, y)

    ax = axes.physical
    x = physical[:, 0]
    y = physical[:, 1]
    ax.set_title("Physical")
    ax.set_xlabel(l_physical_0, fontsize=labelsize)
    ax.set_ylabel(l_physical_1, fontsize=labelsize)
    ax.scatter(x, y, s=s)
    Axis_aspect_2d(ax, 0.8, x, y)

    ax = axes.x
    x = physical[:, 0]
    y = latent_map[:, 0]
    ax.set_title(f"X Correlation = {corr_x:.4f}")
    ax.set_xlabel(l_physical_0, fontsize=labelsize)
    ax.set_ylabel(l_latent_0, fontsize=labelsize)
    ax.scatter(x, y, s=s)
    Axis_aspect_2d(ax, 0.8, x, y)

    ax = axes.y
    x = physical[:, 1]
    y = latent_map[:, 1]
    ax.set_title(f"Y Correlation = {corr_y:.4f}")
    ax.set_xlabel(l_physical_1, fontsize=labelsize)
    ax.set_ylabel(l_latent_1, fontsize=labelsize)
    ax.scatter(x, y, s=s)
    Axis_aspect_2d(ax, 0.8, x, y)

    # fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    correlation()
