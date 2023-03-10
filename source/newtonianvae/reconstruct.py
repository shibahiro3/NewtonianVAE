import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
from torch import Tensor

import models.core
import mypython.plotutil as mpu
import mypython.vision as mv
import tool.util
import view.plot_config
from models.core import NewtonianVAEFamily
from mypython import rdict
from mypython.ai.util import SequenceDataLoader
from mypython.pyutil import Seq, Seq2, add_version
from mypython.terminal import Color, Prompt
from simulation.env import obs2img
from tool import checker, paramsmanager
from tool.util import Preferences
from view.label import Label

view.plot_config.apply()
try:
    import view._plot_config

    view._plot_config.apply()
except:
    pass


def reconstruction_(
    *,
    model: NewtonianVAEFamily,
    batchdata: dict,  # (T, B, D)
    save_path: Optional[str] = None,
):
    with torch.set_grad_enabled(False):

        T = batchdata["action"].shape[0]
        episodes = batchdata["action"].shape[1]

        all_steps = T * episodes
        batchdata["delta"].unsqueeze_(-1)

        model.eval()
        model.init_cache()
        model.is_save = True
        model(batchdata)
        model.convert_cache(type_to="numpy")

        # ============================================================
        plt.rcParams.update(
            {
                "figure.figsize": (7.74, 9.08),
                "figure.subplot.left": 0.05,
                "figure.subplot.right": 0.95,
                "figure.subplot.bottom": 0.05,
                "figure.subplot.top": 0.9,
                "figure.subplot.hspace": 0.4,
                "figure.subplot.wspace": 0.5,
            }
        )

        fig = plt.figure()
        mpu.get_figsize(fig)

        recon_camera_key_list = model.cache["camera"].keys()

        class Ax:
            def __init__(self) -> None:

                n_img = len(recon_camera_key_list)
                r_img = Seq2(n_img, (1, 0), lazy=False)
                rx = Seq2(2, (1, 0), start=r_img.length, lazy=True)
                c_obs_recon = Seq2(3, (3, 0), start=0, lazy=True)

                gs = GridSpec(nrows=r_img.length + rx.length, ncols=c_obs_recon.length)

                self.action = fig.add_subplot(gs[: r_img.length, c_obs_recon.a : c_obs_recon.b])
                c_obs_recon.update()

                self.observations: List[plt.Axes] = []
                self.recons: List[plt.Axes] = []

                for _ in range(n_img):
                    self.observations.append(
                        fig.add_subplot(gs[r_img.a : r_img.b, c_obs_recon.a : c_obs_recon.b])
                    )

                r_img.reset()
                c_obs_recon.update()

                for _ in range(n_img):
                    self.recons.append(
                        fig.add_subplot(gs[r_img.a : r_img.b, c_obs_recon.a : c_obs_recon.b])
                    )

                start = 1
                bar_size = 2
                line_end = c_obs_recon.length - bar_size

                self.x_mean = fig.add_subplot(gs[rx.a : rx.b, start:line_end])
                self.x_mean_bar = fig.add_subplot(gs[rx.a : rx.b, line_end : line_end + bar_size])
                rx.update()
                self.x_std = fig.add_subplot(gs[rx.a : rx.b, start:line_end])
                self.x_std_bar = fig.add_subplot(gs[rx.a : rx.b, line_end : line_end + bar_size])

            def clear(self):
                self.action.clear()
                for ax in self.observations:
                    ax.clear()
                for ax in self.recons:
                    ax.clear()
                self.x_mean.clear()
                self.x_mean_bar.clear()
                self.x_std.clear()
                self.x_std_bar.clear()

        axes = Ax()
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

                # ============================================================
                self.t += 1
                dim_colors = mpu.cmap(model.cell.dim_x, "prism")

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

                color_map = cm.get_cmap("rainbow")

                # ============================================================
                ax = axes.action
                N = model.cell.dim_x
                R = range(1, N + 1)
                ax.set_title(r"$\mathbf{u}_{t-1}$ (Original)")
                # ax.set_xlabel("$u_x$")
                # ax.set_ylabel("$u_y$")
                # ax.set_xlim(-1, 1)
                ax.set_ylim(-1.2, 1.2)
                # ax.arrow(0, 0, action[t, 0], action[t, 1], head_width=0.05)
                ax.bar(
                    R,
                    batchdata["action"][self.t, self.ep].squeeze(0).cpu().numpy(),
                    color=dim_colors,
                    width=0.5,
                )
                ax.set_xticks(R)
                ax.set_xticklabels([str(s) for s in R])
                # ax.tick_params(bottom=False, labelbottom=False)
                mpu.Axis_aspect_2d(ax, 1)

                # ============================================================
                for i, k in enumerate(recon_camera_key_list):
                    ax = axes.observations[i]
                    ax.set_title(r"$\mathbf{I}_t$ " f"({k}, Original)")
                    ax.imshow(obs2img(batchdata["camera"][k][self.t, self.ep].squeeze(0)))
                    ax.set_axis_off()
                    i += 1

                # ============================================================
                for i, k in enumerate(recon_camera_key_list):
                    ax = axes.recons[i]
                    ax.set_title(r"$\mathbf{I}_t$ " f"({k}, Reconstructed)")
                    ax.imshow(obs2img(self.model.cache["camera"][k][self.t, self.ep]))
                    ax.set_axis_off()

                # ============================================================
                ax = axes.x_mean
                N = model.cell.dim_x
                ax.set_title(r"$\mathbf{x}_{1:t}$ (mean)")
                ax.set_xlim(0, T)
                ax.set_ylim(self.min_x_mean, self.max_x_mean)
                # ax.yaxis.set_major_formatter(FormatStrFormatter("%2.2f"))
                for i in range(N):
                    ax.plot(
                        range(self.t),
                        model.cache["x_mean"][: self.t, self.ep, i],
                        color=color_map(1 - i / N),
                        lw=1,
                    )

                # ============================================================
                ax = axes.x_mean_bar
                N = model.cell.dim_x
                R = range(1, N + 1)
                ax.set_title(r"$\mathbf{x}_t$ " f"(mean, dim: {N})")
                ax.set_ylim(self.min_x_mean, self.max_x_mean)
                # ax.yaxis.set_major_formatter(FormatStrFormatter("%2.2f"))
                ax.bar(
                    R,
                    self.model.cache["x_mean"][self.t, self.ep],
                    color=[color_map(1 - i / N) for i in range(N)],
                )
                ax.set_xticks(R)
                ax.set_xticklabels([str(s) for s in R])
                ax.tick_params(left=False, labelleft=False)
                # ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
                mpu.Axis_aspect_2d(ax, 1)

                # ============================================================
                ax = axes.x_std
                N = model.cell.dim_x
                ax.set_title(r"$\mathbf{x}_{1:t}$ (std)")
                ax.set_xlim(0, T)
                ax.set_ylim(self.min_x_std, self.max_x_std)
                # ax.yaxis.set_major_formatter(FormatStrFormatter("%2.2f"))
                for i in range(N):
                    ax.plot(
                        range(self.t),
                        self.model.cache["x_std"][: self.t, self.ep, i],
                        color=color_map(1 - i / N),
                        lw=1,
                    )

                # ============================================================
                ax = axes.x_std_bar
                N = model.cell.dim_x
                R = range(1, N + 1)
                ax.set_title(r"$\mathbf{x}_t$ " f"(std, dim: {N})")
                ax.set_ylim(self.min_x_std, self.max_x_std)
                # ax.yaxis.set_major_formatter(FormatStrFormatter("%2.2f"))
                ax.bar(
                    R,
                    self.model.cache["x_std"][self.t, self.ep],
                    color=[color_map(1 - i / N) for i in range(N)],
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
):
    if save_anim:
        checker.large_episodes(episodes)

    params = paramsmanager.Params(config)

    dtype, device = tool.util.dtype_device(
        dtype=params.test.dtype,
        device=params.test.device,
    )

    model, manage_dir, weight_path, saved_params = tool.util.load(
        root=params.path.saves_dir,
        model_place=models.core,
    )
    model: NewtonianVAEFamily
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
        shuffle=False,
    )
    batchdata = next(testloader)

    if save_anim:
        path_result = tool.util.priority(params.path.results_dir, saved_params.path.results_dir)
        save_path = Path(
            path_result, manage_dir.stem, f"E{weight_path.stem}_reconstructed.{format}"
        )
        save_path = add_version(save_path)
    else:
        save_path = None

    reconstruction_(model=model, batchdata=batchdata, save_path=save_path)

    print()


def _min_max(x, axis=None, pad_ratio=0.2):
    min = np.nanmin(x, axis=axis)
    max = np.nanmax(x, axis=axis)
    padding = (max - min) * pad_ratio
    min -= padding
    max += padding
    return min, max
