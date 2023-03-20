import itertools
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
from _private import unet
from models.core import NewtonianVAEBase
from mypython.ai.util import SequenceDataLoader, swap01
from mypython.pyutil import Seq, add_version
from mypython.terminal import Color
from tool import paramsmanager
from view.label import Label

view.plot_config.apply()
try:
    import view._plot_config

    view._plot_config.apply()
except:
    pass


def correlation_(
    *,
    model: NewtonianVAEBase,
    batchdata: dict,  # (T, B, D)
    save_path: Optional[str] = None,
    show: bool = True,
    all: bool = True,
    position_name: str = "position",
    format: Optional[List[str]] = None,
):
    with torch.no_grad():

        T = batchdata["action"].shape[0]
        episodes = batchdata["action"].shape[1]

        batchdata["delta"].unsqueeze_(-1)

        model.eval()
        model.init_cache()
        model.is_save = True
        model(batchdata)
        model.convert_cache(type_to="numpy")

        position = batchdata[position_name].detach().cpu().numpy()

        dim = model.cache["x"].shape[-1]

        # (B, T, D)
        latent_map = swap01(model.cache["x"])
        physical = swap01(position)

        color = mpu.cmap(episodes, "rainbow")  # per batch color
        # color = ["#377eb880" for _ in range(episodes)]

        # =========================== plt ===========================

        if all:
            plt.rcParams.update(
                {
                    "figure.figsize": (10.51, 8.46),
                    "figure.subplot.left": 0.1,
                    "figure.subplot.right": 0.95,
                    "figure.subplot.bottom": 0.1,
                    "figure.subplot.top": 0.95,
                    "figure.subplot.hspace": 0.4,
                    "figure.subplot.wspace": 0.2,
                    #
                    "lines.marker": "o",
                    "lines.markersize": 1,
                    "lines.markeredgecolor": "None",
                    "lines.linestyle": "None",
                }
            )

            fig, axes = plt.subplots(dim, dim, sharex="col", sharey="row", squeeze=False)
            mpu.get_figsize(fig)

            for ld in range(dim):
                for pd in range(dim):
                    ax = axes[ld][pd]
                    x = physical[..., pd]
                    y = latent_map[..., ld]
                    corr = np.corrcoef(x.reshape(-1), y.reshape(-1))[0, 1]
                    if pd == ld:
                        ax.set_title(f"Correlation = {corr:.4f}", color="red")
                    else:
                        ax.set_title(f"Correlation = {corr:.4f}")

                    for ep in range(episodes):
                        ax.plot(x[ep], y[ep], color=color[ep])

            for pd in range(dim):
                axes[-1][pd].set_xlabel(f"Physical {pd+1}")

            for ld in range(dim):
                axes[ld][0].set_ylabel(f"Latent {ld+1}")

        else:
            plt.rcParams.update(
                {
                    "figure.figsize": (10.5, 4),
                    "figure.subplot.left": 0.1,
                    "figure.subplot.right": 0.95,
                    "figure.subplot.bottom": 0.15,
                    "figure.subplot.top": 0.9,
                    "figure.subplot.hspace": 0.5,
                    "figure.subplot.wspace": 0.3,
                    #
                    "lines.marker": "o",
                    "lines.markersize": 1,
                    "lines.markeredgecolor": "None",
                    "lines.linestyle": "None",
                }
            )

            fig, axes = plt.subplots(1, dim, squeeze=True)
            mpu.get_figsize(fig)

            for d in range(dim):
                x = physical[..., d]
                y = latent_map[..., d]
                corr = np.corrcoef(x.reshape(-1), y.reshape(-1))[0, 1]
                ax = axes[d]
                ax.set_title(f"Correlation = {corr:.4f}")
                ax.set_xlabel(f"Physical {d+1}")
                ax.set_ylabel(f"Latent {d+1}")
                for ep in range(episodes):
                    ax.plot(x[ep], y[ep], color=color[ep])

        # ============================================================
        if show:
            if (save_path is not None) and (format is not None):
                mpu.register_save_path(fig, save_path, format)

            plt.show()
        else:
            if save_path is not None:
                save_path: Path = Path(save_path).with_suffix(".png")
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path)
                Color.print("[correlation] saved to:", save_path)

        plt.clf()
        plt.close()


def correlation(
    config: str,
    episodes: int,
    all: bool,
    format: List[str],
    position_name: str = "position",
):

    params = paramsmanager.Params(config)

    dtype, device = tool.util.dtype_device(
        dtype=params.train.dtype,
        device=params.train.device,
    )

    model, managed_dir, weight_path, saved_params = tool.util.load(
        root=params.path.saves_dir,
        model_place=models.core,
    )
    model: NewtonianVAEBase
    model.type(dtype)
    model.to(device)

    path_data = tool.util.priority(params.path.data_dir, saved_params.path.data_dir)

    testloader = SequenceDataLoader(
        root=Path(path_data, "episodes"),
        start=params.test.data_start,
        stop=params.test.data_stop,
        batch_size=episodes,
        dtype=dtype,
        device=device,
    )
    batchdata = next(testloader)

    if saved_params.others.get("use_unet", False):
        with torch.no_grad():
            pre_unet = unet.MobileUNet(out_channels=1).to(device)

            p_ = Path(params.path.saves_dir, "unet", "weight.pth")

            # p_ = Path(
            #     params.path.saves_dir,
            #     managed_dir.stem,
            #     "unet_with_nvae",
            #     "weight",
            #     weight_path.name,
            # )

            pre_unet.load_state_dict(torch.load(p_))

            pre_unet.eval()

            T, B, C, H, W = batchdata["camera"]["self"].shape
            batchdata["camera"]["self"] = unet.pre(
                pre_unet, batchdata["camera"]["self"].reshape(-1, C, H, W)
            ).reshape(T, B, C, H, W)

    # ============================================================
    path_result = tool.util.priority(params.path.results_dir, saved_params.path.results_dir)
    save_path = Path(path_result, managed_dir.stem, f"E{weight_path.stem}_correlation.pdf")
    save_path = add_version(save_path)

    correlation_(
        model=model,
        batchdata=batchdata,
        all=all,
        save_path=save_path,
        format=format,
        show=True,
        position_name=position_name,
    )
