import common

common.set_path(__file__)

import argparse
import sys
from argparse import RawTextHelpFormatter
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

import models.core
import mypython.plotutil as mpu
import mypython.plt_layout as pltl
import mypython.vision as mv
import tool.preprocess
import tool.util
import view.plot_config

# from _private import unet
from models.core import NewtonianVAEBase
from mypython import rdict
from mypython.ai.util import SequenceDataLoader
from mypython.terminal import Color, Prompt
from simulation.env import obs2img
from tool import checker, paramsmanager


def main():
    # fmt: off
    description = \
"""\
Examples:
  $ python reconstruct.py -c config/reacher2d.json5
"""
    parser = argparse.ArgumentParser(allow_abbrev=False, formatter_class=RawTextHelpFormatter, description=description)
    parser.add_argument("-c", "--config", type=str, required=True, **common.config)
    parser.add_argument("--episodes", type=int, default=10)  # If too large, get torch.cuda.OutOfMemoryError
    parser.add_argument("--save-anim", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--format", type=str, default="mp4", **common.format_video)
    args = parser.parse_args()
    # fmt: on

    reconstruction(**vars(args))


def reconstruction_(
    *,
    model: NewtonianVAEBase,
    batchdata: dict,  # (T, B, D)
    save_path: Optional[str] = None,
):
    with torch.no_grad():

        T = batchdata["action"].shape[0]
        dim_x = batchdata["action"].shape[-1]
        episodes = batchdata["action"].shape[1]

        all_steps = T * episodes
        batchdata["delta"].unsqueeze_(-1)

        model.eval()
        model.init_cache()
        model.is_save = True
        model(batchdata)
        model.convert_cache(type_to="numpy")
        rdict.to_numpy(batchdata)

        # ============================================================
        plt.rcParams.update({"axes.titlesize": 13})

        fig = plt.figure(figsize=(7.74, 9.08))
        mpu.get_figsize(fig)

        camera_names = model.cache["camera"].keys()

        class Ax:
            def __init__(self) -> None:
                fig.subplots_adjust(bottom=0.05)

                n_img = len(camera_names)

                self.action = pltl.Plotter()
                self.observations = [pltl.Plotter() for _ in range(n_img)]
                self.recons = [pltl.Plotter() for _ in range(n_img)]
                self.x_mean = pltl.Plotter(flex=4)
                self.x_mean_bar = pltl.Plotter()
                self.x_std = pltl.Plotter(flex=4)
                self.x_std_bar = pltl.Plotter()

                self.layout = pltl.Column(
                    [
                        pltl.Row(
                            [
                                self.action,
                                pltl.Column(
                                    [
                                        pltl.Row(self.observations, space=0.2),
                                        pltl.Row(self.recons, space=0.2),
                                    ],
                                    space=0.5,
                                    flex=n_img,
                                ),
                            ],
                            flex=1.5,
                        ),
                        pltl.Column(
                            [
                                pltl.Row([self.x_mean, self.x_mean_bar]),
                                pltl.Row([self.x_std, self.x_std_bar]),
                            ],
                            space=0.5,
                            flex=1,
                        ),
                    ],
                )

                pltl.compile(fig, self.layout)

            def clear(self):
                pltl.clear(self.layout)

        axes = Ax()
        colors_action = mpu.cmap(dim_x, "prism")
        colors_latent = mpu.cmap(dim_x, "rainbow")
        # ============================================================

        class AnimPack:
            def __init__(self) -> None:

                self.t = -1

                self.model = model
                self.model.is_save = True

            def init(self):

                # self.one_batchdata: Dict[str, Tensor] = {}
                # for k, v in batchdata.items():
                #     self.one_batchdata[k] = v[:, [self.episode_cnt]]
                # self.one_batchdata["delta"].unsqueeze_(-1)

                # self.model.init_cache()
                # self.model(self.one_batchdata)
                # self.model.convert_cache(type_to="numpy", treat_batch="squeeze")

                self.min_x_mean, self.max_x_mean = _min_max(self.model.cache["x_mean"])
                self.min_x_std, self.max_x_std = _min_max(self.model.cache["x_std"])

            def anim_func(self, frame_cnt):
                axes.clear()

                Prompt.print_one_line(
                    f"{frame_cnt+1:5d} / {all_steps} ({(frame_cnt+1)*100/all_steps:.1f} %) "
                )

                mod = frame_cnt % T
                if mod == 0:
                    self.t = -1
                    self.ep = frame_cnt // T + mod  # episode count

                    self.action_min = batchdata["action"][:, self.ep].min()
                    self.action_max = batchdata["action"][:, self.ep].max()

                # ============================================================
                self.t += 1

                if frame_cnt == -1:
                    self.t = T - 1
                    self.init()

                if self.t == 0:
                    self.init()

                # print(self.ep, self.t)

                fig.suptitle(
                    f"Test episode: {self.ep+1}, t = {self.t:3d}",
                    fontname="monospace",
                )

                # ============================================================
                ax = axes.action.ax
                N = dim_x
                R = range(1, N + 1)
                ax.set_title(r"$\mathbf{u}_{t-1}$ (Original)")
                # ax.set_xlabel("$u_x$")
                # ax.set_ylabel("$u_y$")
                # ax.set_xlim(-1, 1)
                pad = (self.action_max - self.action_min) * 0.1
                ax.set_ylim(self.action_min - pad, self.action_max + pad)
                # ax.arrow(0, 0, action[t, 0], action[t, 1], head_width=0.05)
                ax.bar(
                    R,
                    batchdata["action"][self.t, self.ep],
                    color=colors_action,
                    width=0.5,
                )
                ax.set_xticks(R)
                ax.set_xticklabels([str(s) for s in R])
                # ax.tick_params(bottom=False, labelbottom=False)
                mpu.Axis_aspect_2d(ax, 1)

                # ============================================================
                for i, k in enumerate(camera_names):
                    ax = axes.observations[i].ax
                    ax.set_title(r"$\mathbf{I}_t$ " f"({k}, Original)")
                    ax.imshow(obs2img(batchdata["camera"][k][self.t, self.ep]))
                    ax.set_axis_off()

                # ============================================================
                for i, k in enumerate(camera_names):
                    ax = axes.recons[i].ax
                    ax.set_title(r"$\mathbf{I}_t$ " f"({k}, Reconstructed)")
                    ax.imshow(obs2img(self.model.cache["camera"][k][self.t, self.ep]))
                    ax.set_axis_off()

                # ============================================================
                ax = axes.x_mean.ax
                N = dim_x
                ax.set_title(r"$\mathbf{x}_{1:t}$ (mean)")
                ax.set_xlim(0, T)
                ax.set_ylim(self.min_x_mean, self.max_x_mean)
                # ax.yaxis.set_major_formatter(FormatStrFormatter("%2.2f"))
                for i in range(N):
                    ax.plot(
                        range(self.t),
                        model.cache["x_mean"][: self.t, self.ep, i],
                        color=colors_latent[i],
                        lw=1,
                    )

                # ============================================================
                ax = axes.x_mean_bar.ax
                N = dim_x
                R = range(1, N + 1)
                ax.set_title(r"$\mathbf{x}_t$ " f"(mean, dim: {N})")
                ax.set_ylim(self.min_x_mean, self.max_x_mean)
                # ax.yaxis.set_major_formatter(FormatStrFormatter("%2.2f"))
                ax.bar(
                    R,
                    self.model.cache["x_mean"][self.t, self.ep],
                    color=colors_latent,
                )
                ax.set_xticks(R)
                ax.set_xticklabels([str(s) for s in R])
                ax.tick_params(left=False, labelleft=False)
                # ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
                mpu.Axis_aspect_2d(ax, 1)

                # ============================================================
                ax = axes.x_std.ax
                N = dim_x
                ax.set_title(r"$\mathbf{x}_{1:t}$ (std)")
                ax.set_xlim(0, T)
                ax.set_ylim(self.min_x_std, self.max_x_std)
                # ax.yaxis.set_major_formatter(FormatStrFormatter("%2.2f"))
                for i in range(N):
                    ax.plot(
                        range(self.t),
                        self.model.cache["x_std"][: self.t, self.ep, i],
                        color=colors_latent[i],
                        lw=1,
                    )

                # ============================================================
                ax = axes.x_std_bar.ax
                N = dim_x
                R = range(1, N + 1)
                ax.set_title(r"$\mathbf{x}_t$ " f"(std, dim: {N})")
                ax.set_ylim(self.min_x_std, self.max_x_std)
                # ax.yaxis.set_major_formatter(FormatStrFormatter("%2.2f"))
                ax.bar(
                    R,
                    self.model.cache["x_std"][self.t, self.ep],
                    color=colors_latent,
                )
                ax.set_xticks(R)
                ax.set_xticklabels([str(s) for s in R])
                ax.tick_params(left=False, labelleft=False)
                # ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
                mpu.Axis_aspect_2d(ax, 1)

        p = AnimPack()

        mpu.anim_mode(
            "save" if save_path is not None else "anim",
            fig,
            p.anim_func,
            all_steps,
            interval=40,
            freeze_cnt=-1,
            save_path=save_path,
        )

        model.init_cache()
        model.is_save = False

        plt.clf()
        plt.close()


def reconstruction(
    config: str,
    episodes: int,
    save_anim: bool,
    format: str,
    shuffle=False,
):
    if save_anim:
        checker.large_episodes(episodes)

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

    batchdata = SequenceDataLoader(
        patterns=params.test.path,
        batch_size=episodes,
        dtype=dtype,
        device=device,
        shuffle=shuffle,
        preprocess=getattr(tool.preprocess, saved_params.others.get("preprocess", ""), None),
    ).sample_batch(verbose=True)

    if save_anim:
        path_result = params.path.results_dir
        save_path = tool.util.save_pathname(
            root=path_result,
            day_time=managed_dir.stem,
            epoch=weight_path.stem,
            descr="reconstructed",
            format=format,
        )
    else:
        save_path = None

    # if saved_params.others.get("use_unet", False):
    #     with torch.no_grad():
    #         pre_unet = unet.MobileUNet(out_channels=1).to(device)
    #         p_ = Path(params.path.saves_dir, "unet", "weight.pth")
    #         pre_unet.load_state_dict(torch.load(p_))
    #         pre_unet.eval()
    #         T, B, C, H, W = batchdata["camera"]["self"].shape
    #         batchdata["camera"]["self"] = unet.pre(
    #             pre_unet, batchdata["camera"]["self"].reshape(-1, C, H, W)
    #         ).reshape(T, B, C, H, W)

    view.plot_config.apply()
    reconstruction_(model=model, batchdata=batchdata, save_path=save_path)

    print()


def _min_max(x, axis=None, pad_ratio=0.2):
    min = np.nanmin(x, axis=axis)
    max = np.nanmax(x, axis=axis)
    padding = (max - min) * pad_ratio
    min -= padding
    max += padding
    return min, max


if __name__ == "__main__":
    main()
