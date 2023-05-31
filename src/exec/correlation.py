import common

common.set_path(__file__)

import argparse
import io
import sys
from argparse import RawTextHelpFormatter
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Optional, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from natsort import natsorted
from PIL import Image

import models.core
import mypython.plotutil as mpu
import mypython.vision as mv
import tool.preprocess
import tool.util
import view.plot_config
from models.core import NewtonianVAEBase
from mypython import rdict
from mypython.ai.train import save_pathname
from mypython.ai.util import SequenceDataLoader, swap01
from mypython.pyutil import add_version
from mypython.terminal import Color, Prompt
from tool import paramsmanager, prepost
from tool.util import create_prepostprocess
from unet_mask.seg_data import mask_unet


def main():
    # fmt: off
    description = \
"""\
Show correlation using test data
  Save: Press S key on a window
"""
    parser = argparse.ArgumentParser(allow_abbrev=False, formatter_class=RawTextHelpFormatter, description=description)
    parser.add_argument("-c", "--config", type=str, required=True, **common.config)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--all", action="store_true", help="Show correlations for all combinations")
    parser.add_argument("--position-name", type=str, default="position")
    parser.add_argument("--all-epochs", action="store_true")
    parser.add_argument("--no-window", action='store_true', help="Save instantly without displaying a window")
    # parser.add_argument("--format", type=str, default=["svg", "pdf", "png"], nargs="*", **common.format_file) # Deprecated
    parser.add_argument("--format", type=str, default=common.default_fig_formats, nargs="*")
    args = parser.parse_args()
    # fmt: on

    correlation(**vars(args))


def print_corr(corrs):
    corrs = np.array(corrs)

    if corrs.ndim == 1:
        if (np.abs(corrs) > 0.9).all():
            Color.print(
                Prompt.del_line + "Correlations:", [f"{corr:.4f}" for corr in corrs], c=Color.red
            )
        elif (np.abs(corrs) > 0.8).all():
            Color.print(
                Prompt.del_line + "Correlations:", [f"{corr:.4f}" for corr in corrs], c=Color.green
            )
        else:
            print(Prompt.del_line + "Correlations:", [f"{corr:.4f}" for corr in corrs])
    else:
        print("Correlations:")
        print(corrs)


def correlation_cal(
    *,
    model: NewtonianVAEBase,
    batchdata: dict,  # (T, B, D)
    all: bool = True,
    position_name: str = "position",
):
    with torch.no_grad():

        T, B, dim = batchdata["action"].shape
        # dim = model.dim_x

        model.eval()

        model.init_cache()
        model.is_save = True
        model(batchdata)  # TODO: support List[batchdata]   dataloader.sample_batch
        model.convert_cache(type_to="numpy")
        x = model.cache["x"]
        # xs = [] # TODO: same

        position = batchdata[position_name].detach().cpu().numpy()

        # to (B, T, D)
        latent_map = swap01(x)
        physical = swap01(position)

        if all:
            corr_arr = np.empty((dim, dim))
            for ld in range(dim):
                for pd in range(dim):
                    x = physical[..., pd]
                    y = latent_map[..., ld]
                    corr_arr[ld][pd] = np.corrcoef(x.reshape(-1), y.reshape(-1))[0, 1]
            # print_corr(np.diag(corr_arr))
            # print(corr_arr)

        else:
            corr_arr = np.empty((dim,))
            for d in range(dim):
                x = physical[..., d]
                y = latent_map[..., d]
                corr_arr[d] = np.corrcoef(x.reshape(-1), y.reshape(-1))[0, 1]
            # print_corr(corr_arr)

    return corr_arr, physical, latent_map


def correlation_(
    *,
    model: NewtonianVAEBase,
    batchdata: dict,  # (T, B, D)
    save_path: Optional[str] = None,
    fig_mode: str = "show",
    all: bool = True,
    position_name: str = "position",
    format: Optional[List[str]] = None,
    title=None,
):
    assert fig_mode in ("show", "save", "numpy")

    # DO NOT do preprocess of batchdata here -> upper layer code

    corr_arr, physical, latent_map = correlation_cal(
        model=model, batchdata=batchdata, all=all, position_name=position_name
    )
    print_corr(corr_arr)

    T, B = batchdata["action"].shape[:2]
    dim = model.dim_x

    color = mpu.cmap(B, "rainbow")  # per batch color
    # color = ["#377eb880" for _ in range(episodes)]

    view.plot_config.apply()
    plt.rcParams.update({"axes.titlesize": 20})

    corrs = []
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

        if title is not None:
            plt.rcParams.update({"figure.subplot.top": 0.9})

        fig, axes = plt.subplots(dim, dim, sharex="col", sharey="row", squeeze=False)
        mpu.get_figsize(fig)

        for ld in range(dim):
            for pd in range(dim):
                ax = axes[ld][pd]
                x = physical[..., pd]
                y = latent_map[..., ld]
                corr = corr_arr[ld][pd]
                if pd == ld:
                    ax.set_title(f"Correlation = {corr:.4f}", color="red")
                    assert len(corrs) == pd
                    corrs.append(corr)
                else:
                    ax.set_title(f"Correlation = {corr:.4f}")

                for ep in range(B):
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

        if title is not None:
            plt.rcParams.update({"figure.subplot.top": 0.85})

        fig, axes = plt.subplots(1, dim, squeeze=True)
        mpu.get_figsize(fig)

        for d in range(dim):
            x = physical[..., d]
            y = latent_map[..., d]
            corr = corr_arr[d]
            corrs.append(corr)
            ax = axes[d]
            ax.set_title(f"Correlation = {corr:.4f}")
            ax.set_xlabel(f"Physical {d+1}")
            ax.set_ylabel(f"Latent {d+1}")
            for ep in range(B):
                ax.plot(x[ep], y[ep], color=color[ep])

    # ============================================================

    figarray = None
    if title is not None:
        fig.suptitle(title)

    if fig_mode == "show":
        assert save_path is not None
        assert format is not None
        mpu.register_save_path(fig, save_path, format)
        plt.show()

    elif fig_mode == "save":
        assert save_path is not None
        assert format is not None
        save_path: Path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        for fmt in format:
            p = save_path.with_suffix(f".{fmt}")
            p = add_version(p)
            plt.savefig(p)
            Color.print("saved to:", p)

    elif fig_mode == "numpy":
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        figarray = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        figarray = cv2.imdecode(figarray, 1)
        figarray = cv2.cvtColor(figarray, cv2.COLOR_BGR2RGB)

    plt.clf()
    plt.close()

    return figarray


def correlation(
    config: str,
    all: bool,  # all axis
    format: List[str],
    all_epochs=False,
    episodes: int = None,
    position_name: str = "position",
    no_window: bool = False,
):
    params = paramsmanager.Params(config)

    dtype, device = tool.util.dtype_device(
        dtype=params.test.dtype,
        device=params.test.device,
    )

    model, managed_dir, weight_path, saved_params = tool.util.load(
        root=params.path.saves_dir,
        model_place=models.core,
    )
    model: NewtonianVAEBase
    model.type(dtype)
    model.to(device)

    print("weight path:", weight_path)

    preprocess, postprocesses = create_prepostprocess(params, device=device)
    keypaths = params.others.get("keypaths", None)

    batchdata = SequenceDataLoader(
        patterns=params.test.path,
        batch_size=episodes,
        dtype=dtype,
        device=device,
        preprocess=preprocess,
        keypaths=keypaths,
    ).sample_batch(verbose=True)

    batchdata["delta"].unsqueeze_(-1)

    # if saved_params.others.get("use_unet", False):
    #     with torch.no_grad():
    #         pre_unet = unet.MobileUNet(out_channels=1).to(device)

    #         p_ = Path(params.path.saves_dir, "unet", "weight.pth")

    #         # p_ = Path(
    #         #     params.path.saves_dir,
    #         #     managed_dir.stem,
    #         #     "unet_with_nvae",
    #         #     "weight",
    #         #     weight_path.name,
    #         # )

    #         pre_unet.load_state_dict(torch.load(p_))

    #         pre_unet.eval()

    #         T, B, C, H, W = batchdata["camera"]["self"].shape
    #         batchdata["camera"]["self"] = unet.pre(
    #             pre_unet, batchdata["camera"]["self"].reshape(-1, C, H, W)
    #         ).reshape(T, B, C, H, W)

    # ============================================================
    path_result = params.path.results_dir

    if not all_epochs:
        save_path = save_pathname(
            root=path_result, day_time=managed_dir.stem, epoch=weight_path.stem, descr="correlation"
        )

        correlation_(
            model=model,
            batchdata=batchdata,
            all=all,
            save_path=save_path,
            format=format,
            fig_mode="save" if no_window else "show",
            position_name=position_name,
        )
    else:
        figarrays = []
        f_weights = list(weight_path.parent.glob("*"))
        weight_paths = natsorted(weight_paths)
        for fw in f_weights:
            print(fw)
            model.load_state_dict(torch.load(fw))
            model.type(dtype)
            model.to(device)

            figarray = correlation_(
                model=model,
                batchdata=batchdata,
                all=all,
                save_path=fw,
                format=format,
                fig_mode="numpy",
                position_name=position_name,
                title=f"Epoch: {fw.stem}",
            )
            figarrays.append(figarray)

        save_path = save_pathname(
            root=path_result, day_time=managed_dir.stem, descr="correlation_epochs"
        ).with_suffix(".gif")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        Color.print("saving to:", save_path)
        figarrays = [Image.fromarray(img) for img in figarrays]
        figarrays[0].save(
            save_path, save_all=True, append_images=figarrays[1:], duration=400, loop=0
        )


if __name__ == "__main__":
    main()
