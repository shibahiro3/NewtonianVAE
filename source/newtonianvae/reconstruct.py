import argparse
import sys
from pathlib import Path
from typing import Dict, Union

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
import tool.util
from models.core import NewtonianVAE, NewtonianVAEV2
from mypython.pyutil import Seq, add_version
from mypython.terminal import Color, Prompt
from simulation.env import obs2img
from tool import argset, checker, paramsmanager
from tool.dataloader import DataLoader
from tool.paramsmanager import Params, ParamsEval
from tool.util import Preferences
from view.label import Label

try:
    import tool._plot_config

    figsize = tool._plot_config.figsize_reconstruct
except:
    figsize = None


parser = argparse.ArgumentParser(allow_abbrev=False)
argset.episodes(parser)
argset.cf(parser)
argset.path_data(parser, required=False)
argset.path_model(parser, required=False)
argset.path_result(parser, required=False)
argset.save_anim(parser)
argset.fix_xmap_size(parser, required=False)
_args = parser.parse_args()


class Args:
    episodes = _args.episodes
    cf = _args.cf
    path_model = _args.path_model
    path_data = _args.path_data
    path_result = _args.path_result
    fix_xmap_size = _args.fix_xmap_size
    save_anim = _args.save_anim


args = Args()


def reconstruction():
    # ============================================================
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9)
    mpu.get_figsize(fig)

    class Ax:
        def __init__(self) -> None:
            gs = GridSpec(nrows=6, ncols=6, hspace=0.4, wspace=0.5)
            up = 2

            self.action = fig.add_subplot(gs[:up, 0:2])
            self.observation = fig.add_subplot(gs[:up, 2:4])
            self.reconstructed = fig.add_subplot(gs[:up, 4:6])

            self.x_mean = fig.add_subplot(gs[up, :4])
            self.x_dims = fig.add_subplot(gs[up, 4])
            self.x_map = fig.add_subplot(gs[up, 5])

            self.v = fig.add_subplot(gs[up + 1, :4])
            self.v_dims = fig.add_subplot(gs[up + 1, 4])

            self.xhat_mean = fig.add_subplot(gs[up + 2, :4])
            self.xhat_mean_dims = fig.add_subplot(gs[up + 2, 4])

            self.xhat_std = fig.add_subplot(gs[up + 3, :4])
            self.xhat_std_dims = fig.add_subplot(gs[up + 3, 4])

        def clear(self):
            for ax in self.__dict__.values():
                ax.clear()

    axes = Ax()
    # ============================================================

    if args.save_anim == "save":
        checker.large_episodes(args.episodes)

    torch.set_grad_enabled(False)

    _params = paramsmanager.Params(args.cf)
    params_eval = _params.eval
    path_model = tool.util.priority(args.path_model, _params.external.save_path)
    path_data = tool.util.priority(args.path_data, _params.external.data_path)
    # data_path = get_path_data(args.path_data, params)
    del _params
    path_result = tool.util.priority(args.path_result, params_eval.result_path)

    dtype, device = tool.util.dtype_device(
        dtype=params_eval.dtype,
        device=params_eval.device,
    )

    testloader = DataLoader(
        root=Path(path_data, "episodes"),
        start=params_eval.data_start,
        stop=params_eval.data_stop,
        batch_size=args.episodes,
        dtype=dtype,
        device=device,
    )
    action, observation, delta, position = next(testloader)

    model, manage_dir, weight_path, params = tool.util.load(
        root=path_model, model_place=models.core
    )

    model: Union[NewtonianVAE, NewtonianVAEV2]
    model.type(dtype)
    model.to(device)
    model.train(params_eval.training)
    model.is_save = True

    all_steps = params.train.max_time_length * args.episodes

    class AnimPack:
        def __init__(self) -> None:

            self.t = -1

            self.model = model
            self.model.is_save = True

        def _attach_nan(self, x, i: int):
            xp = np
            return xp.concatenate(
                [
                    x[: self.t, i],
                    xp.full(
                        (params.train.max_time_length - self.t,),
                        xp.nan,
                        dtype=x.dtype,
                    ),
                ]
            )

        def init(self):

            self.action, self.observation, self.delta = (
                action[:, [self.episode_cnt]],
                observation[:, [self.episode_cnt]],
                delta[:, [self.episode_cnt]],
            )

            self.model(action=self.action, observation=self.observation, delta=self.delta)
            self.model.LOG2numpy(squeezeN1=True)

            # Color.print(self.action.shape)
            # Color.print(self.observation.shape)
            # Color.print(self.delta.shape)

            # ----------------
            self.min_x_mean, self.max_x_mean = _min_max(self.model.LOG_x_mean)

            if args.fix_xmap_size is None:
                self.min_x_map, self.max_x_map = _min_max(self.model.LOG_x_mean, axis=0)
            else:
                l_ = args.fix_xmap_size
                self.min_x_map, self.max_x_map = np.array([[-l_, -l_], [l_, l_]])

            self.min_v, self.max_v = _min_max(self.model.LOG_v)

            if "xhat" in model.cell.info:
                self.min_xhat_mean, self.max_xhat_mean = _min_max(self.model.LOG_xhat_mean)
                self.min_xhat_std, self.max_xhat_std = _min_max(self.model.LOG_xhat_std)

        def anim_func(self, frame_cnt):
            axes.clear()

            Prompt.print_one_line(
                f"{frame_cnt+1:5d} / {all_steps} ({(frame_cnt+1)*100/all_steps:.1f} %)"
            )

            mod = frame_cnt % params.train.max_time_length
            if mod == 0:
                self.t = -1
                self.episode_cnt = frame_cnt // params.train.max_time_length + mod

            # ======================================================
            self.t += 1
            color_action = mpu.cmap(model.cell.dim_x, "prism")

            if frame_cnt == -1:
                self.episode_cnt = np.random.randint(0, args.episodes)
                self.t = params.train.max_time_length - 1
                self.init()

            if self.t == 0:
                self.init()

            fig.suptitle(
                f"Validational episode: {self.episode_cnt+1}, t = {self.t:3d}",
                fontname="monospace",
            )

            color_map = cm.get_cmap("rainbow")

            # ===============================================================
            ax = axes.action
            ax.set_title(r"$\mathbf{u}_{t-1}$ (Original)")
            # ax.set_xlabel("$u_x$")
            # ax.set_ylabel("$u_y$")
            # ax.set_xlim(-1, 1)
            ax.set_ylim(-1.2, 1.2)
            # ax.arrow(0, 0, action[t, 0], action[t, 1], head_width=0.05)
            ax.bar(
                range(model.cell.dim_x),
                self.action[self.t].squeeze(0).cpu().numpy(),
                color=color_action,
                width=0.5,
            )
            ax.tick_params(bottom=False, labelbottom=False)
            mpu.Axis_aspect_2d(ax, 1)

            # ===============================================================
            ax = axes.observation
            ax.set_title(r"$\mathbf{I}_t$ (Original)")
            ax.imshow(obs2img(self.observation[self.t].squeeze(0)))
            ax.set_axis_off()

            # ===============================================================
            ax = axes.reconstructed
            ax.set_title(r"$\mathbf{I}_t$ (Reconstructed)")
            ax.imshow(obs2img(self.model.LOG_I_dec[self.t]))
            ax.set_axis_off()

            # ===============================================================
            ax = axes.x_mean
            N = model.cell.dim_x
            ax.set_title(r"$\mathbf{x}_{1:t}$")
            ax.set_xlim(0, params.train.max_time_length)
            ax.set_ylim(self.min_x_mean, self.max_x_mean)
            ax.yaxis.set_major_formatter(FormatStrFormatter("%2.2f"))
            for i in range(N):
                ax.plot(
                    range(params.train.max_time_length),
                    self._attach_nan(self.model.LOG_x_mean, i),
                    color=color_map(1 - i / N),
                    lw=1,
                )

            # ===============================================================
            ax = axes.x_dims
            N = model.cell.dim_x
            ax.set_title(r"$\mathbf{x}_{t}$  " f"(dim: {N})")
            ax.set_ylim(self.min_x_mean, self.max_x_mean)
            # ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%2.2f"))
            ax.bar(
                range(N),
                self.model.LOG_x_mean[self.t],
                color=[color_map(1 - i / N) for i in range(N)],
            )
            ax.tick_params(bottom=False, labelbottom=False)
            mpu.Axis_aspect_2d(ax, 1)

            # ===============================================================
            ax = axes.x_map
            ax.set_title(r"$\mathbf{x}_{1:t}$")
            ax.set_xlim(self.min_x_map[0], self.max_x_map[0])
            ax.set_ylim(self.min_x_map[1], self.max_x_map[1])
            ax.set_aspect(1)
            # ax.set_xticks([self.min_x_mean, self.max_x_mean])
            # ax.set_yticks([self.min_x_mean, self.max_x_mean])
            # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            # ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.plot(
                self._attach_nan(self.model.LOG_x_mean, 0),
                self._attach_nan(self.model.LOG_x_mean, 1),
                marker="o",
                ms=2,
            )
            ax.plot(
                self.model.LOG_x_mean[self.t, 0],
                self.model.LOG_x_mean[self.t, 1],
                marker="o",
                ms=5,
                color="red",
            )

            # ===============================================================
            ax = axes.v
            N = model.cell.dim_x
            ax.set_title(r"$\mathbf{v}_{1:t}$")
            ax.set_xlim(0, params.train.max_time_length)
            ax.set_ylim(self.min_v, self.max_v)
            ax.yaxis.set_major_formatter(FormatStrFormatter("%2.2f"))
            for i in range(N):
                ax.plot(
                    range(params.train.max_time_length),
                    self._attach_nan(self.model.LOG_v, i),
                    color=color_map(1 - i / N),
                    lw=1,
                )

            # ===============================================================
            ax = axes.v_dims
            N = model.cell.dim_x
            ax.set_title(r"$\mathbf{v}_t$  " f"(dim: {N})")
            ax.yaxis.set_major_formatter(FormatStrFormatter("%2.2f"))
            # ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_ylim(self.min_v, self.max_v)
            ax.bar(
                range(N),
                self.model.LOG_v[self.t],
                color=[color_map(1 - i / N) for i in range(N)],
            )
            ax.tick_params(bottom=False, labelbottom=False)
            mpu.Axis_aspect_2d(ax, 1)

            if "xhat" in model.cell.info:
                # ===============================================================
                ax = axes.xhat_mean
                N = model.cell.dim_xhat
                ax.set_title(r"$\hat{\mathbf{x}}_{1:t}$")
                ax.set_xlim(0, params.train.max_time_length)
                ax.set_ylim(self.min_xhat_mean, self.max_xhat_mean)
                ax.yaxis.set_major_formatter(FormatStrFormatter("%2.2f"))
                for i in range(N):
                    ax.plot(
                        range(params.train.max_time_length),
                        self._attach_nan(self.model.LOG_xhat_mean, i),
                        color=color_map(1 - i / N),
                        lw=1,
                    )

                # ===============================================================
                ax = axes.xhat_mean_dims
                N = model.cell.dim_xhat
                ax.set_title(r"$\hat{\mathbf{x}}_t$  " f"(dim: {N})")
                ax.yaxis.set_major_formatter(FormatStrFormatter("%2.2f"))
                ax.set_ylim(self.min_xhat_mean, self.max_xhat_mean)
                ax.bar(
                    range(N),
                    self.model.LOG_xhat_mean[self.t],
                    color=[color_map(1 - i / N) for i in range(N)],
                )
                ax.tick_params(bottom=False, labelbottom=False)
                mpu.Axis_aspect_2d(ax, 1)

                # ===============================================================
                ax = axes.xhat_std
                N = model.cell.dim_xhat
                ax.set_title(r"std of $\hat{\mathbf{x}}_{1:t}$")
                ax.set_xlim(0, params.train.max_time_length)
                ax.set_ylim(self.min_xhat_std, self.max_xhat_std)
                ax.yaxis.set_major_formatter(FormatStrFormatter("%2.2f"))
                for i in range(N):
                    ax.plot(
                        range(params.train.max_time_length),
                        self._attach_nan(self.model.LOG_xhat_std, i),
                        color=color_map(1 - i / N),
                        lw=1,
                    )

                # ===============================================================
                ax = axes.xhat_std_dims
                N = model.cell.dim_xhat
                ax.set_title(r"std of $\hat{\mathbf{x}}_t$  " f"(dim: {N})")
                ax.set_ylim(self.min_xhat_std, self.max_xhat_std)
                ax.yaxis.set_major_formatter(FormatStrFormatter("%2.2f"))
                ax.bar(
                    range(N),
                    self.model.LOG_xhat_std[self.t],
                    color=[color_map(1 - i / N) for i in range(N)],
                )
                ax.tick_params(bottom=False, labelbottom=False)
                mpu.Axis_aspect_2d(ax, 1)

    p = AnimPack()

    save_path = Path(path_result, f"{manage_dir.stem}_W{weight_path.stem}_reconstructed.mp4")
    save_path = add_version(save_path)
    mpu.anim_mode(
        "save" if args.save_anim else "anim",
        fig,
        p.anim_func,
        all_steps,
        interval=40,
        freeze_cnt=-1,
        save_path=save_path,
    )

    print()


def _min_max(x, axis=None, pad_ratio=0.2):
    min = np.nanmin(x, axis=axis)
    max = np.nanmax(x, axis=axis)
    padding = (max - min) * pad_ratio
    min -= padding
    max += padding
    return min, max


if __name__ == "__main__":
    reconstruction()
