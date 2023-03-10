import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from torch import Tensor
from typing_extensions import Self

import mypython.error as merror
import mypython.plotutil as mpu
import mypython.vision as mv
import view.plot_config
from mypython import rdict
from mypython.ai.util import SequenceDataLoader, to_np
from mypython.terminal import Color, Prompt
from simulation.env import obs2img
from tool import paramsmanager

view.plot_config.apply()
try:
    import view._plot_config

    view._plot_config.apply()
except:
    pass


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

        gs = GridSpec(nrows=1, ncols=n_cam + 2)
        self.ax_action = self.fig.add_subplot(gs[0, 0])
        self.ax_cameras: List[plt.Axes] = []
        for i in range(1, 1 + n_cam):
            self.ax_cameras.append(self.fig.add_subplot(gs[0, i]))

        if env_domain == "point_mass_3d":
            self.ax_position = self.fig.add_subplot(gs[0, n_cam + 1], projection="3d")
        else:
            self.ax_position = self.fig.add_subplot(gs[0, n_cam + 1])

    def axes_clear(self):
        self.ax_action.clear()
        for ax in self.ax_cameras:
            ax.clear()
        # self.ax_position.clear()

    def first_episode(self):
        # self.t = 0
        # self.episode_cnt += 1

        if self.env_domain == "point_mass_3d":
            self.ax_position.clear()

    def frame(
        self,
        t: int,
        episode_cnt: int,
        action: np.ndarray,
        Ist: Dict[str, Tensor],
        position: np.ndarray,
        position_title: Optional[str] = None,
        set_lim: Callable[[Self], None] = None,
    ):
        self.axes_clear()

        action = to_np(action)

        self.fig.suptitle(
            f"Episode: {episode_cnt}, t = {t:3d}",
            fontname="monospace",
        )

        dim_colors = mpu.cmap(len(action), "prism")

        # ==================================================
        ax = self.ax_action
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
            ax = self.ax_cameras[_i]
            ax.set_title("$\mathbf{I}_t$" f" ({k})")
            ax.imshow(obs2img(v))
            ax.set_xlabel(f"{v.shape[-1]} px")
            ax.set_ylabel(f"{v.shape[-2]} px")
            ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
            _i += 1

        # ==================================================
        ax = self.ax_position
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

        if callable(set_lim):
            set_lim(self)

        mpu.Axis_aspect_2d(self.ax_action, 1)
        mpu.Axis_aspect_2d(self.ax_position, 1)


def main(
    config: str,
    episodes: int,
    save_anim: bool,
    format: str,
):
    params = paramsmanager.Params(config)

    merror.check_dir(params.path.data_dir)

    batchdata = SequenceDataLoader(
        root=Path(params.path.data_dir, "episodes"),
        start=params.train.data_start,
        stop=params.train.data_stop,
        batch_size=episodes,
        dtype=torch.float32,
        show_selected_index=True,
        # shuffle=False,
    ).sample_batch(verbose=True)

    # rdict.to_numpy(batchdata)

    action = batchdata["action"]
    position = batchdata["position"]
    # position = batchdata["relative_position"]

    T = action.shape[0]
    all_steps = T * episodes

    plt_render = ShowData(
        win_title="Show Data",
        n_cam=len(batchdata["camera"]),
        # env_domain=env.domain,
        # env_task=env.task,
        # env_position_wrap=env.position_wrap,
    )

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

            plt_render.frame(
                t=self.t + 1,
                episode_cnt=self.episode_cnt + 1,
                action=action[self.t, self.episode_cnt],
                Ist={k: v[self.t, self.episode_cnt] for k, v in batchdata["camera"].items()},
                position=position[self.t, self.episode_cnt],
                set_lim=lambda p: (
                    p.ax_action.set_ylim(action.min() - 0.1, action.max() + 0.1),
                    p.ax_position.set_ylim(position.min() - 0.1, position.max() + 0.1),
                ),
            )

    p = AnimPack()
    mpu.anim_mode(
        "save" if save_anim else "anim",
        plt_render.fig,
        p.anim_func,
        all_steps,
        interval=40,
        save_path=Path(params.path.data_dir, f"data.{format}"),
    )
