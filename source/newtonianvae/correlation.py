import sys
from pathlib import Path
from typing import Dict, List, Union

import classopt
import json5
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter

import models.core
import mypython.plotutil as mpu
import mypython.vision as mv
import tool.plot_config
import tool.util
from models.core import NewtonianVAEFamily
from mypython.ai.util import SequenceDataLoader, swap01
from mypython.pyutil import Seq
from mypython.terminal import Color
from tool import argset, paramsmanager
from view.label import Label

tool.plot_config.apply()
try:
    import tool._plot_config

    tool._plot_config.apply()
except:
    pass

config = {
    "figure.figsize": (8.89, 5.83),
    "figure.subplot.left": 0.1,
    "figure.subplot.right": 0.95,
    "figure.subplot.bottom": 0.05,
    "figure.subplot.top": 0.98,
    "figure.subplot.hspace": 0.2,
    "figure.subplot.wspace": 0.5,
    #
    "lines.marker": "o",
    "lines.markersize": 1,
    "lines.markeredgecolor": "None",
    "lines.linestyle": "None",
}
plt.rcParams.update(config)


@classopt.classopt(default_long=True, default_short=False)
class Args:
    config: str = classopt.config(**argset.descr_config, required=True)
    episodes: int = classopt.config(**argset.descr_episodes, required=True)
    path_model: str = classopt.config(**argset.descr_path_model, required=False)
    path_data: str = classopt.config(**argset.descr_path_data, required=False)
    path_result: str = classopt.config(**argset.descr_path_result, required=False)
    fix_xmap_size: float = classopt.config(metavar="S", help="xmap size")
    env_domain: str = classopt.config(metavar="ENV")
    format: List[str] = classopt.config(nargs="*", default=["svg", "pdf"])


args = Args.from_args()  # pylint: disable=E1101


def correlation():
    # ============================================================
    fig = plt.figure()
    mpu.get_figsize(fig)

    class Ax:
        def __init__(self) -> None:
            gs = GridSpec(nrows=2, ncols=3)
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

    _params = paramsmanager.Params(args.config)
    params_eval = _params.eval
    path_model = tool.util.priority(args.path_model, _params.external.save_path)
    path_data = tool.util.priority(args.path_data, _params.external.data_path)
    del _params
    path_result = tool.util.priority(args.path_result, params_eval.result_path)

    dtype, device = tool.util.dtype_device(
        dtype=params_eval.dtype,
        device=params_eval.device,
    )

    testloader = SequenceDataLoader(
        root=Path(path_data, "episodes"),
        names=["action", "observation", "delta", "position"],
        start=params_eval.data_start,
        stop=params_eval.data_stop,
        batch_size=args.episodes,
        dtype=dtype,
        device=device,
    )
    action, observation, delta, position = next(testloader)
    delta.unsqueeze_(-1)
    position = position.detach().cpu()

    model, manage_dir, weight_path, params = tool.util.load(
        root=path_model, model_place=models.core
    )

    model: NewtonianVAEFamily
    model.type(dtype)
    model.to(device)
    model.train(params_eval.training)
    model.is_save = True

    model(action=action, observation=observation, delta=delta)
    model.LOG2numpy()

    # (BS, T, D)
    latent_map = swap01(model.LOG_x)
    physical = swap01(position)

    corr_p0l0 = np.corrcoef(
        physical.reshape(-1, model.cell.dim_x)[:, 0], latent_map.reshape(-1, model.cell.dim_x)[:, 0]
    )[0, 1]
    corr_p1l1 = np.corrcoef(
        physical.reshape(-1, model.cell.dim_x)[:, 1], latent_map.reshape(-1, model.cell.dim_x)[:, 1]
    )[0, 1]
    corr_p0l1 = np.corrcoef(
        physical.reshape(-1, model.cell.dim_x)[:, 0], latent_map.reshape(-1, model.cell.dim_x)[:, 1]
    )[0, 1]
    corr_p1l0 = np.corrcoef(
        physical.reshape(-1, model.cell.dim_x)[:, 1], latent_map.reshape(-1, model.cell.dim_x)[:, 0]
    )[0, 1]

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
    save_path = Path(path_result, f"{manage_dir.stem}_W{weight_path.stem}_correlation.pdf")
    # save_path = add_version(save_path)
    mpu.register_save_path(fig, save_path, args.format)

    plt.show()


if __name__ == "__main__":
    correlation()
