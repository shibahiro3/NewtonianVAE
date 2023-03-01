import pickle
import shutil
import sys
import time
from collections import ChainMap
from numbers import Number
from pathlib import Path
from typing import Dict, List, Sequence, Type, Union, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from matplotlib.gridspec import GridSpec
from torch import Tensor

import mypython.plotutil as mpu
import mypython.vision as mv
import tool.util
import view.plot_config
from mypython import rdict
from mypython.terminal import Color, Prompt
from simulation.env import ControlSuiteEnvWrap, obs2img
from tool import checker, paramsmanager
from tool.util import Preferences

view.plot_config.apply()
try:
    import view._plot_config

    view._plot_config.apply()
except:
    pass


class Collector:
    def __init__(self) -> None:
        self.data = {}

    def capture(self, data):
        pass

    def save(self, path_dir):
        path_dir = Path(path_dir)
        path_dir.mkdir(parents=True, exist_ok=True)
        # for k, v in self.data.items():
        #     np.save(Path(path_dir, f"{k}.npy"), self.action)


def main(
    config: str,
    episodes: int,
    watch: str,
    save_anim: bool,
    format: str,
):
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

    if save_anim and watch != "plt":
        Color.print(
            "Ignore --save-anim option: Use --watch=plt option to save videos", c=Color.coral
        )

    if save_anim and watch == "plt":
        checker.large_episodes(episodes)

    params = paramsmanager.Params(config)

    env = ControlSuiteEnvWrap(**params.raw["ControlSuiteEnvWrap"])
    T = env.max_episode_length // env.action_repeat
    all_steps = T * episodes

    print("observation size:", env.observation_size)
    print("action size:", env.action_size)
    print("action range:", env.action_range)

    data_path = Path(params.path.data_dir)
    if watch is None:
        if len(list(data_path.glob("episodes/*"))) > 0:
            print(
                f'\n"{data_path}" already has data. This directory will be erased and replaced with new data.'
            )
            if input("Do you want to continue? [y/n] ") != "y":
                print("Abort.")
                return
            shutil.rmtree(data_path)

        data_path.mkdir(parents=True, exist_ok=True)
        params.save_simenv(Path(data_path, "params_env_bk.json5"))
        Preferences.put(data_path, "id", int(time.time() * 1000))

    if watch == "plt":

        def on_close(event):
            sys.exit()

        fig = plt.figure()
        mpu.get_figsize(fig)
        fig.canvas.mpl_connect("close_event", on_close)

        class Ax:
            def __init__(self) -> None:
                observations = env.reset()
                # print(observations.keys())
                n_cam = len([s for s in observations["camera"].keys()])
                # print(n_cam)

                gs = GridSpec(nrows=1, ncols=n_cam + 2)
                self.action = fig.add_subplot(gs[0, 0])
                self.cameras: List[plt.Axes] = []
                for i in range(1, 1 + n_cam):
                    self.cameras.append(fig.add_subplot(gs[0, i]))

                if env.domain == "point_mass_3d":
                    self.position = fig.add_subplot(gs[0, n_cam + 1], projection="3d")
                else:
                    self.position = fig.add_subplot(gs[0, n_cam + 1])
                # self.position = fig.add_subplot(gs[0, n_cam + 1])

            def clear(self):
                self.action.clear()
                for ax in self.cameras:
                    ax.clear()
                # self.position.clear()

        axes = Ax()

    class AnimPack:
        def __init__(self) -> None:
            self.collector = Collector()

            self.t = 0
            self.episode_cnt = 0

        def anim_func(self, frame_cnt):
            if save_anim:
                Prompt.print_one_line(
                    f"{frame_cnt+1:5d} / {all_steps} ({(frame_cnt+1)*100/all_steps:.1f} %) "
                )

            mod = frame_cnt % T
            if mod == 0:
                env.reset()

                self.collector = Collector()
                self.episode_data = {}

                self.t = 0
                self.episode_cnt = frame_cnt // T + mod

                if env.domain == "point_mass_3d" and watch == "plt":
                    axes.position.clear()

            # ======================================================

            self.t += 1

            ### core ###
            action = env.sample_random_action()
            # action = env.zeros_action()
            observations, done = env.step(action)

            step_data = observations
            step_data["action"] = action
            step_data["delta"] = 0.1
            step_data["relative_position"] = step_data["position"] - step_data["target_position"]

            rdict.to_numpy(step_data, ignore_scalar=True)
            rdict.append_a_to_b(step_data, self.episode_data)
            # rdict.show(step_data, "step_data")
            # rdict.show(self.episode_data, "episode_data")

            if watch == "render":
                env.render()

            elif watch == "plt":
                axes.clear()

                fig.suptitle(
                    f"episode: {self.episode_cnt+1}, t = {self.t:3d}",
                    fontname="monospace",
                )

                dim_colors = mpu.cmap(len(action), "prism")

                # ==================================================
                ax = axes.action
                ax.set_title(r"$\mathbf{u}_{t-1}$")
                ax.set_ylim(-1.2, 1.2)
                ax.bar(range(len(action)), action, color=dim_colors, width=0.5)
                mpu.Axis_aspect_2d(ax, 1)
                ax.set_xticks(range(len(action)))
                if env.domain == "reacher2d":
                    ax.set_xlabel("Torque")
                    ax.set_xticklabels([r"$\mathbf{u}[1]$ (shoulder)", r"$\mathbf{u}[2]$ (wrist)"])
                elif env.domain == "point_mass" and env.task == "easy":
                    ax.set_xticklabels([r"$\mathbf{u}[1]$ (x)", r"$\mathbf{u}[2]$ (y)"])
                elif env.domain == "point_mass_3d" and env.task == "easy":
                    ax.set_xticklabels(
                        [r"$\mathbf{u}[1]$ (x)", r"$\mathbf{u}[2]$ (y)", r"$\mathbf{u}[3]$ (z)"]
                    )

                # ==================================================
                _i = 0
                for k, v in observations["camera"].items():
                    ax = axes.cameras[_i]
                    ax.set_title("$\mathbf{I}_t$" f" ({k})")
                    ax.imshow(obs2img(v))
                    ax.set_xlabel(f"{v.shape[-1]} px")
                    ax.set_ylabel(f"{v.shape[-2]} px")
                    ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
                    _i += 1

                # ==================================================
                ax = axes.position
                ax.set_title("Position")
                position = observations["position"]

                if env.domain == "reacher2d" and env.position_wrap == "None":
                    axes.position.clear()
                    ax.set_ylim(-np.pi, np.pi)
                    ax.bar(range(len(position)), position, color=dim_colors, width=0.5)
                    mpu.Axis_aspect_2d(ax, 1)
                    ax.set_xlabel("Angle")
                    ax.set_xticks(range(len(position)))
                    ax.set_xticklabels([r"$\theta_1$ (shoulder)", r"$\theta_2$ (wrist)"])

                elif (
                    env.domain == "reacher2d" and env.position_wrap == "endeffector"
                ) or env.domain == "point_mass":
                    axes.position.clear()
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(position[0], position[1], marker="o", ms=10, color="orange")
                    mpu.cartesian_coordinate(ax, 0.35)  # dm_control wall: 0.3

                elif env.domain == "point_mass_3d":
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
                    axes.position.clear()
                    ax.bar(range(len(position)), position, color=dim_colors, width=0.5)
                    mpu.Axis_aspect_2d(ax, 1)

            if done and not save_anim:

                # rdict.to_numpy(self.episode_data)
                # rdict.show(self.episode_data, "episode_data (save)")

                if watch is None:
                    episodes_dir = Path(data_path, "episodes")
                    episodes_dir.mkdir(parents=True, exist_ok=True)

                    rdict.to_numpy(self.episode_data)
                    # rdict.show(self.episode_data, "episode_data (save)")
                    with open(Path(episodes_dir, f"{self.episode_cnt}.pickle"), "wb") as f:
                        pickle.dump(self.episode_data, f)

                    info = Color.green + "saved" + Color.reset
                else:
                    info = Color.coral + "not saved" + Color.reset

                Prompt.print_one_line(f"episode: {self.episode_cnt+1}, T = {self.t}  {info} ")

    p = AnimPack()

    if watch == "plt":
        save_path = Path(data_path, f"data.{format}")
        mpu.anim_mode(
            "save" if save_anim else "anim",
            fig,
            p.anim_func,
            T * episodes,
            interval=40,
            save_path=save_path,
        )

    else:
        for frame_cnt in range(T * episodes):
            p.anim_func(frame_cnt)

    print()


if __name__ == "__main__":
    main()
