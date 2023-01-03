import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

import mypython.plotutil as mpu
import mypython.vision as mv
import tool.util
import view.plot_config
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
        self.action = []
        self.observation = []
        self.delta = []
        self.position = []

    def save(self, path_dir):
        path_dir = Path(path_dir)
        path_dir.mkdir(parents=True, exist_ok=True)
        np.save(Path(path_dir, "action.npy"), self.action)
        np.save(Path(path_dir, "observation.npy"), self.observation)
        np.save(Path(path_dir, "delta.npy"), self.delta)
        np.save(Path(path_dir, "position.npy"), self.position)


def main(
    config: str,
    episodes: int,
    watch: str,
    save_anim: bool,
    format: str,
):
    plt.rcParams.update(
        {
            "figure.figsize": (10.14, 4.05),
            "figure.subplot.left": 0.05,
            "figure.subplot.right": 0.95,
            "figure.subplot.bottom": 0,
            "figure.subplot.top": 1,
            "figure.subplot.wspace": 0.3,
        }
    )

    if save_anim and watch != "plt":
        Color.print(
            "Ignore --save-anim option: Use --watch=plt option to save videos", c=Color.coral
        )

    if save_anim and watch == "plt":
        checker.large_episodes(episodes)

    params = paramsmanager.Params(config)

    env = ControlSuiteEnvWrap(**params.raw_["ControlSuiteEnvWrap"])
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
                gs = GridSpec(nrows=1, ncols=3)
                self.action = fig.add_subplot(gs[0, 0])
                self.observation = fig.add_subplot(gs[0, 1])
                self.position = fig.add_subplot(gs[0, 2])

            def clear(self):
                for ax in self.__dict__.values():
                    ax.clear()

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
                self.collector = Collector()

                env.reset()
                self.t = 0
                self.episode_cnt = frame_cnt // T + mod

            # ======================================================

            self.t += 1

            ### core ###
            action = env.sample_random_action()
            # action = env.zeros_action()
            observation, _, done, position = env.step(action)

            # print("===")
            # print("action shape:     ", tuple(action.shape))  # (2,)
            # print("observation shape:", tuple(observation.shape))  # (3, 64, 64)
            # print("position shape:   ", position.shape)  # (2,)
            self.collector.action.append(action.numpy())
            self.collector.observation.append(observation.numpy())
            self.collector.delta.append(0.1)
            self.collector.position.append(position)
            ############

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

                # ==================================================
                ax = axes.observation
                ax.set_title("$\mathbf{I}_t$")
                ax.imshow(obs2img(observation))

                # ==================================================
                ax = axes.position
                ax.set_title("Position")
                if env.domain == "reacher2d" and env.position_wrap == "None":
                    ax.set_ylim(-np.pi, np.pi)
                    ax.bar(range(len(position)), position, color=dim_colors, width=0.5)
                    mpu.Axis_aspect_2d(ax, 1)
                    ax.set_xlabel("Angle")
                    ax.set_xticks(range(len(position)))
                    ax.set_xticklabels([r"$\theta_1$ (shoulder)", r"$\theta_2$ (wrist)"])
                elif (
                    env.domain == "reacher2d" and env.position_wrap == "endeffector"
                ) or env.domain == "point_mass":
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(position[0], position[1], marker="o", ms=10, color="orange")
                    mpu.cartesian_coordinate(ax, 0.35)  # dm_control wall: 0.3

            if done and not save_anim:
                print(f"episode: {self.episode_cnt+1}, T = {self.t}")

                if watch is None:
                    episode_dir = Path(data_path, "episodes", f"{self.episode_cnt}")
                    self.collector.save(episode_dir)
                    Color.print("saved", c=Color.green)
                else:
                    Color.print("not saved", c=Color.coral)

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


if __name__ == "__main__":
    main()
