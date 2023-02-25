import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter

import models.core
import mypython.plotutil as mpu
import mypython.vision as mv
import tool.util
import view.plot_config
from models.core import NewtonianVAEFamily
from mypython.ai.util import SequenceDataLoader, swap01
from mypython.pyutil import Seq
from mypython.terminal import Color
from tool import paramsmanager
from view.label import Label

view.plot_config.apply()
try:
    import view._plot_config

    view._plot_config.apply()
except:
    pass


def correlation(
    config: str,
    episodes: int,
    fix_xmap_size: float,
    env_domain: str,
    format: List[str],
):
    # ============================================================
    plt.rcParams.update(
        {
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
    )

    fig = plt.figure()
    mpu.get_figsize(fig)

    class Ax:
        def __init__(self) -> None:
            dim = 3
            gs = GridSpec(nrows=2, ncols=1 + dim)
            v1 = Seq()
            # self.physical = fig.add_subplot(gs[0, 0])
            self.latent_map = fig.add_subplot(gs[0:2, 0])

            self.p0l0 = fig.add_subplot(gs[0, 1])
            self.p1l1 = fig.add_subplot(gs[0, 2])
            self.p2l2 = fig.add_subplot(gs[0, 3])

            self.p0l1 = fig.add_subplot(gs[1, 1])
            self.p1l0 = fig.add_subplot(gs[1, 2])

    axes = Ax()
    label = Label(env_domain)
    # ============================================================

    # =========================== load ===========================
    torch.set_grad_enabled(False)

    params = paramsmanager.Params(config)

    dtype, device = tool.util.dtype_device(
        dtype=params.train.dtype,
        device=params.train.device,
    )

    model, manage_dir, weight_path, saved_params = tool.util.load(
        root=params.path.saves_dir,
        model_place=models.core,
    )
    model: NewtonianVAEFamily
    model.type(dtype)
    model.to(device)
    model.eval()
    model.is_save = True

    path_data = tool.util.priority(params.path.data_dir, saved_params.path.data_dir)
    path_result = tool.util.priority(params.path.results_dir, saved_params.path.results_dir)

    testloader = SequenceDataLoader(
        root=Path(path_data, "episodes"),
        start=params.eval.data_start,
        stop=params.eval.data_stop,
        batch_size=episodes,
        dtype=dtype,
        device=device,
    )

    del params
    del saved_params
    # ======================== end of load =======================

    batchdata = next(testloader)
    batchdata["delta"].unsqueeze_(-1)
    position = batchdata["position"].detach().cpu()
    # position = batchdata["relative_position"].detach().cpu()

    print("Calculating...")
    model(batchdata)
    model.convert_cache(type_to="numpy")

    # (B, T, D)
    latent_map = swap01(model.cache["x"])
    physical = swap01(position)

    corr_p0l0 = np.corrcoef(
        physical.reshape(-1, model.cell.dim_x)[:, 0],
        latent_map.reshape(-1, model.cell.dim_x)[:, 0],
    )[0, 1]
    corr_p1l1 = np.corrcoef(
        physical.reshape(-1, model.cell.dim_x)[:, 1],
        latent_map.reshape(-1, model.cell.dim_x)[:, 1],
    )[0, 1]

    corr_p2l2 = None
    if model.cell.dim_x >= 3:
        corr_p2l2 = np.corrcoef(
            physical.reshape(-1, model.cell.dim_x)[:, 2],
            latent_map.reshape(-1, model.cell.dim_x)[:, 2],
        )[0, 1]

    # ===

    corr_p0l1 = np.corrcoef(
        physical.reshape(-1, model.cell.dim_x)[:, 0],
        latent_map.reshape(-1, model.cell.dim_x)[:, 1],
    )[0, 1]
    corr_p1l0 = np.corrcoef(
        physical.reshape(-1, model.cell.dim_x)[:, 1],
        latent_map.reshape(-1, model.cell.dim_x)[:, 0],
    )[0, 1]
    print("Done")

    # ============================================================

    color = mpu.cmap(episodes, "rainbow")  # per batch color
    # color = ["#377eb880" for _ in range(episodes)]

    lmax = fix_xmap_size

    # ============================================================
    x = latent_map[..., 0]
    y = latent_map[..., 1]

    ax = axes.latent_map
    ax.set_title("Latent map")

    for i in range(episodes):
        ax.plot(x[i], y[i], color=color[i])
    label.set_axes_L0L1(ax, lmax)

    # ============================================================
    x = physical[..., 0]
    y = latent_map[..., 0]

    ax = axes.p0l0
    ax.set_title(f"Correlation = {corr_p0l0:.4f}")

    for i in range(episodes):
        ax.plot(x[i], y[i], color=color[i])
    label.set_axes_P0L0(ax, lmax)

    # ============================================================
    x = physical[..., 1]
    y = latent_map[..., 1]

    ax = axes.p1l1
    ax.set_title(f"Correlation = {corr_p1l1:.4f}")

    for i in range(episodes):
        ax.plot(x[i], y[i], color=color[i])
    label.set_axes_P1L1(ax, lmax)

    # ============================================================
    if model.cell.dim_x >= 3:
        x = physical[..., 2]
        y = latent_map[..., 2]

        ax = axes.p2l2
        ax.set_title(f"Correlation = {corr_p2l2:.4f}")

        for i in range(episodes):
            ax.plot(x[i], y[i], color=color[i])
        label.set_axes_P2L2(ax, lmax)

    # ============================================================
    x = physical[..., 0]
    y = latent_map[..., 1]

    ax = axes.p0l1
    ax.set_title(f"Correlation = {corr_p0l1:.4f}")

    for i in range(episodes):
        ax.plot(x[i], y[i], color=color[i])
    label.set_axes_P0L1(ax, lmax)

    # ============================================================
    x = physical[..., 1]
    y = latent_map[..., 0]

    ax = axes.p1l0
    ax.set_title(f"Correlation = {corr_p1l0:.4f}")

    for i in range(episodes):
        ax.plot(x[i], y[i], color=color[i])
    label.set_axes_P1L0(ax, lmax)

    # ============================================================
    save_path = Path(path_result, f"{manage_dir.stem}_W{weight_path.stem}_correlation.pdf")
    # save_path = add_version(save_path)
    mpu.register_save_path(fig, save_path, format)

    plt.show()
