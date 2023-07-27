#!/usr/bin/env python3


import argparse
from argparse import RawTextHelpFormatter
from typing import Callable, Dict, List, Optional, Sequence, Type, Union

import common
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from typing_extensions import Self

import mypython.plotutil as mpu
import mypython.plt_layout as pltl
import view.plot_config
from models.mobile_unet import Masker
from mypython import rdict
from mypython.ai.util import SequenceDataLoader, to_numpy
from mypython.terminal import Color, Prompt
from tool import paramsmanager, prepost
from tool.util import create_prepostprocess
from unet_mask.seg_data import mask_unet


def main():
    # fmt: off
    description = \
"""\
Show datasets video from file
"""
    parser = argparse.ArgumentParser(allow_abbrev=False, formatter_class=RawTextHelpFormatter, description=description)
    parser.add_argument("-c", "--config", type=str, required=True, **common.config)
    parser.add_argument("--episodes", type=int, required=True, help="If nor specified, it will load all as batch")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--data-type", type=str, default="train", choices=("train", "valid", "test"))
    parser.add_argument("--env", type=str)
    parser.add_argument("--position-name", type=str, default="position")
    parser.add_argument("--position-title", type=str, default="Position")
    parser.add_argument("--save-path", type=str)
    args = parser.parse_args()
    # fmt: on

    show_data(**vars(args))


class ShowData:
    def __init__(
        self,
        win_title: str,
        n_cam: int,
        env_domain: Optional[str] = None,
        env_task: Optional[str] = None,
        env_position_wrap: Optional[str] = None,
    ) -> None:
        self.n_cam = n_cam
        self.env_domain = env_domain
        self.env_task = env_task
        self.env_position_wrap = env_position_wrap

        view.plot_config.apply()
        plt.rcParams.update(
            {
                "figure.figsize": (11.63, 3.89),
                "figure.subplot.left": 0.05,
                "figure.subplot.right": 0.95,
                "figure.subplot.bottom": 0,
                "figure.subplot.top": 1,
                "figure.subplot.wspace": 0.4,
            }
        )

        # def on_close(event):
        #     sys.exit()

        self.fig = plt.figure(win_title)
        mpu.get_figsize(self.fig)
        # self.fig.canvas.mpl_connect("close_event", on_close)

        self.ax_action = pltl.Plotter()
        self.ax_cameras = [pltl.Plotter() for _ in range(n_cam)]

        if env_domain == "point_mass_3d":
            self.ax_position = pltl.Plotter(clearable=False, projection="3d")
        else:
            self.ax_position = pltl.Plotter()

        self.layout = pltl.compile(
            self.fig,
            pltl.Row(
                # [
                #     self.ax_action,
                #     pltl.Row(self.ax_cameras),
                #     self.ax_position,
                # ]
                [self.ax_action]
                + self.ax_cameras
                + [self.ax_position]
            ),
        )

    def axes_clear(self):
        pltl.clear(self.layout)

    def first_episode(self):
        # self.t = 0
        # self.episode_cnt += 1

        if self.env_domain == "point_mass_3d":
            self.ax_position.ax.clear()

    def frame(
        self,
        t: int,
        episode_cnt: int,
        action: np.ndarray,
        Ist: Dict[str, Tensor],
        position: np.ndarray,
        position_title: Optional[str] = None,
        set_lim_fn: Callable[[Self], None] = None,
    ):
        self.axes_clear()

        action = to_numpy(action)

        self.fig.suptitle(
            f"Episode: {episode_cnt}, t = {t:3d}",
            fontname="monospace",
        )

        dim_colors = mpu.cmap(len(action), "rainbow")

        # ==================================================
        ax = self.ax_action.ax
        R = range(1, len(action) + 1)
        ax.set_title(r"$\mathbf{u}_{t-1}$")
        ax.bar(R, action, color=dim_colors, width=0.5)

        ax.set_xticks(R)
        if self.env_domain == "reacher2d":
            ax.set_xlabel("Torque")
            ax.set_xticklabels([r"$\mathbf{u}[1]$ (shoulder)", r"$\mathbf{u}[2]$ (wrist)"])
        elif self.env_domain == "point_mass" and self.env_task == "easy":
            ax.set_xticklabels([r"$\mathbf{u}[1]$ (x)", r"$\mathbf{u}[2]$ (y)"])
        elif self.env_domain == "point_mass_3d" and self.env_task == "easy":
            ax.set_xticklabels(
                [r"$\mathbf{u}[1]$ (x)", r"$\mathbf{u}[2]$ (y)", r"$\mathbf{u}[3]$ (z)"]
            )

        # ==================================================
        _i = 0
        for k, v in Ist.items():
            ax = self.ax_cameras[_i].ax
            ax.set_title("$\mathbf{I}_t$" f" ({k})")
            ax.imshow(v)  # H W RGB
            ax.set_xlabel(f"{v.shape[-2]} px")  # W
            ax.set_ylabel(f"{v.shape[-3]} px")  # H
            ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
            _i += 1

        # ==================================================
        ax = self.ax_position.ax
        R = range(1, len(position) + 1)

        if not self.env_domain == "point_mass_3d":
            ax.clear()

        if position_title is not None:
            ax.set_title(position_title)

        if self.env_domain == "reacher2d" and self.env_position_wrap == "None":
            ax.set_ylim(-np.pi, np.pi)
            ax.bar(R, position, color=dim_colors, width=0.5)
            ax.set_xlabel("Angle")
            ax.set_xticks(R)
            ax.set_xticklabels([r"$\theta_1$ (shoulder)", r"$\theta_2$ (wrist)"])

        elif (
            self.env_domain == "reacher2d" and self.env_position_wrap == "endeffector"
        ) or self.env_domain == "point_mass":
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.plot(position[0], position[1], marker="o", ms=10, color="orange")
            mpu.cartesian_coordinate(ax, 0.35)  # dm_control wall: 0.3

        elif self.env_domain == "point_mass_3d":
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.plot(
                position[0],
                position[1],
                position[2],
                marker="o",
                ms=3,
                color="orange",
            )
            wall = 0.3
            ax.set_xlim(-wall, wall)
            ax.set_ylim(-wall, wall)
            ax.set_zlim(-wall, wall)

        else:
            ax.bar(R, position, color=dim_colors, width=0.5)

        if set_lim_fn is not None:
            set_lim_fn(self)

        mpu.Axis_aspect_2d(self.ax_action.ax, 1)
        if self.env_domain != "point_mass_3d":
            mpu.Axis_aspect_2d(self.ax_position.ax, 1)


def show_data(
    config: str,
    episodes: int,
    shuffle: bool,
    data_type: bool = "train",
    env: str = None,
    save_path: Optional[str] = None,
    position_name: str = None,
    position_title: str = None,
):
    torch.set_grad_enabled(False)

    assert data_type in ("train", "valid", "test")

    params = paramsmanager.Params(config)

    if position_name is None:
        position_name = "position"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preprocess, postprocesses = create_prepostprocess(params, device=device)
    keypaths = params.others.get("keypaths", None)

    batchdata = SequenceDataLoader(
        patterns=getattr(params, data_type).path,
        batch_size=episodes,
        max_time=params.train.max_time_length,
        dtype=torch.float32,
        device=device,
        shuffle=shuffle,
        preprocess=preprocess,
        keypaths=keypaths,
        random_start=False
        # load_all=True,
    ).sample_batch(verbose=True)

    rdict.to_numpy(batchdata)

    action = batchdata["action"]
    position = batchdata[position_name]

    T = action.shape[0]
    all_steps = T * episodes

    if type(env) == str:
        domain, task = env.split("-")
    else:
        domain, task = None, None

    plt_render = ShowData(
        win_title="Show Data",
        n_cam=len(batchdata["camera"]),
        env_domain=domain,
        env_task=task,
        # env_position_wrap=env.position_wrap,
    )

    if position_title is None:
        if position_name == "position":
            position_title = "Position"
        elif position_name == "relative_position":
            position_title = "Relative Position"

    class AnimPack:
        def __init__(self) -> None:
            self.t = 0
            self.episode_cnt = 0

        def anim_func(self, frame_cnt):
            Prompt.print_one_line(
                f"{frame_cnt+1:5d} / {all_steps} ({(frame_cnt+1)*100/all_steps:.1f} %) "
            )

            mod = frame_cnt % T
            if mod == 0:
                self.t = 0
                self.episode_cnt = frame_cnt // T + mod
                plt_render.first_episode()

            else:
                self.t += 1

            def cam_process(cam):
                # めっちゃ速なった
                cam_ = cam[self.t, self.episode_cnt]
                cam_ = postprocesses.image(cam_)
                return cam_

            plt_render.frame(
                t=self.t + 1,
                episode_cnt=self.episode_cnt + 1,
                action=action[self.t, self.episode_cnt],
                Ist=rdict.apply(batchdata["camera"], cam_process),
                position=position[self.t, self.episode_cnt],
                set_lim_fn=lambda p: (
                    p.ax_action.ax.set_ylim(action.min() - 0.1, action.max() + 0.1),
                    p.ax_position.ax.set_ylim(position.min() - 0.1, position.max() + 0.1),
                ),
                position_title=position_title,
            )

    p = AnimPack()
    mpu.anim_mode(
        "save" if (save_path is not None) else "anim",
        plt_render.fig,
        p.anim_func,
        all_steps,
        interval=40,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
