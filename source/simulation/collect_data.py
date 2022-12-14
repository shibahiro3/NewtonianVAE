import argparse
import shutil
import sys
import time
from pathlib import Path

import json5
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

import mypython.plotutil as mpu
import mypython.vision as mv
import tool.util
from env import ControlSuiteEnvWrap, obs2img
from mypython.terminal import Color, Prompt
from tool import argset, checker, paramsmanager
from tool.util import Preferences

try:
    import tool._plot_config

    figsize = tool._plot_config.figsize_collect
except:
    figsize = None


parser = argparse.ArgumentParser(allow_abbrev=False)
argset.cf(parser)
argset.watch(parser)
argset.episodes(parser)
argset.save_anim(parser)
_args = parser.parse_args()


class Args:
    watch = _args.watch
    episodes = _args.episodes
    save_anim = _args.save_anim
    cf = _args.cf


args = Args()


def env_test():
    if args.save_anim and args.watch != "plt":
        Color.print(
            "Ignore --save-anim option: Use --watch=plt option to save videos", c=Color.coral
        )

    if args.save_anim and args.watch == "plt":
        checker.large_episodes(args.episodes)

    params = paramsmanager.Params(args.cf)

    env = ControlSuiteEnvWrap(**json5.load(open(args.cf))["ControlSuiteEnvWrap"])
    T = env.max_episode_length // env.action_repeat
    all_steps = T * args.episodes

    print("observation size:", env.observation_size)
    print("action size:", env.action_size)
    print("action range:", env.action_range)

    data_path = Path(params.external.data_path)
    if args.watch is None:
        if len(list(data_path.glob("episodes/*"))) > 0:
            print(f'\n"{data_path}" directory will be rewritten.')
            if input("Do you want to continue? [y/n] ") != "y":
                print("Abort.")
                return
            shutil.rmtree(data_path)

        data_path.mkdir(parents=True, exist_ok=True)
        params.save_simenv(Path(data_path, "params_env_bk.json5"))
        Preferences.put(data_path, "id", int(time.time() * 1000))

    if args.watch == "plt":

        def on_close(event):
            sys.exit()

        fig = plt.figure(figsize=figsize)
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0, top=1)
        mpu.get_figsize(fig)
        fig.canvas.mpl_connect("close_event", on_close)

        class Ax:
            def __init__(self) -> None:
                gs = GridSpec(nrows=1, ncols=3, wspace=0.3)
                self.action = fig.add_subplot(gs[0, 0])
                self.observation = fig.add_subplot(gs[0, 1])
                self.position = fig.add_subplot(gs[0, 2])

            def clear(self):
                for ax in self.__dict__.values():
                    ax.clear()

        axes = Ax()

    class AnimPack:
        def __init__(self) -> None:
            self.init_LOG()

            self.t = 0
            self.episode_cnt = 0

        def init_LOG(self):
            self.LOG_action = []
            self.LOG_observation = []
            self.LOG_delta = []
            self.LOG_position = []

        def save_LOG(self, path_dir):
            np.save(Path(path_dir, "action.npy"), self.LOG_action)
            np.save(Path(path_dir, "observation.npy"), self.LOG_observation)
            np.save(Path(path_dir, "delta.npy"), self.LOG_delta)
            np.save(Path(path_dir, "position.npy"), self.LOG_position)

        def anim_func(self, frame_cnt):
            if args.save_anim:
                Prompt.print_one_line(
                    f"{frame_cnt+1:5d} / {all_steps} ({(frame_cnt+1)*100/all_steps:.1f} %)"
                )

            mod = frame_cnt % T
            if mod == 0:
                self.init_LOG()

                env.reset()
                self.t = 0
                self.episode_cnt = frame_cnt // T + mod

            # ======================================================

            self.t += 1

            ### core ###
            action = env.sample_random_action()
            # action = env.zeros_action()
            observation, _, done, position = env.step(action)
            ###---------
            # print("==========")
            # print(action.shape)  # (2,)
            # print(observation.shape)  # (3, 64, 64)
            # print(position.shape)  # (2,)
            self.LOG_action.append(action.numpy())
            self.LOG_observation.append(observation.numpy())
            self.LOG_delta.append(0.1)
            self.LOG_position.append(position)
            ############

            if args.watch == "render":
                env.render()

            elif args.watch == "plt":
                axes.clear()

                fig.suptitle(
                    f"episode: {self.episode_cnt+1}, t = {self.t:3d}",
                    fontname="monospace",
                )

                color_action = mpu.cmap(len(action), "prism")

                # ==================================================
                ax = axes.action
                ax.set_title(r"$\mathbf{u}_{t-1}$")
                ax.set_ylim(-1.2, 1.2)
                ax.bar(range(len(action)), action, color=color_action, width=0.5)
                mpu.Axis_aspect_2d(ax, 1)
                ax.set_xticks(range(len(action)))
                if env.domain == "reacher" or env.domain == "reacher2d":
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
                if (
                    env.domain == "reacher" or env.domain == "reacher2d"
                ) and env.position_wrap == "None":
                    ax.set_ylim(-np.pi, np.pi)
                    ax.bar(range(len(position)), position, color=color_action, width=0.5)
                    mpu.Axis_aspect_2d(ax, 1)
                    ax.set_xlabel("Angle")
                    ax.set_xticks(range(len(position)))
                    ax.set_xticklabels([r"$\theta_1$ (shoulder)", r"$\theta_2$ (wrist)"])
                elif (
                    (env.domain == "reacher" or env.domain == "reacher2d")
                    and env.position_wrap == "endeffector"
                ) or env.domain == "point_mass":
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(position[0], position[1], marker="o", ms=10, color="orange")
                    mpu.cartesian_coordinate(ax, 0.35)

            if done and not args.save_anim:
                print(f"episode: {self.episode_cnt+1}, T = {self.t}")

                if args.watch is None:
                    episode_dir = Path(data_path, "episodes", f"{self.episode_cnt}")
                    episode_dir.mkdir(parents=True, exist_ok=True)
                    self.save_LOG(episode_dir)
                    Color.print("saved", c=Color.green)
                else:
                    Color.print("not saved", c=Color.coral)

    p = AnimPack()

    if args.watch == "plt":
        save_path = Path(data_path, f"data.mp4")
        mpu.anim_mode(
            "save" if args.save_anim else "anim",
            fig,
            p.anim_func,
            T * args.episodes,
            interval=40,
            save_path=save_path,
        )

    else:
        for frame_cnt in range(T * args.episodes):
            p.anim_func(frame_cnt)


if __name__ == "__main__":
    env_test()
