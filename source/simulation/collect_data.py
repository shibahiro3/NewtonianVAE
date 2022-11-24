import argparse
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

import mypython.plotutil as mpu
import mypython.vision as mv
import tool.util
from env import ControlSuiteEnvWrap, obs2img
from mypython.plotutil import cmap
from mypython.terminal import Color
from tool import argset, checker
from tool.params import Params, ParamsSimEnv
from tool.util import backup

try:
    import tool._plot_config

    figsize = tool._plot_config.figsize_collect
except:
    figsize = None


parser = argparse.ArgumentParser(allow_abbrev=False)
argset.watch(parser)
argset.episodes(parser)
argset.save_anim(parser)
argset.cf_simenv(parser)
argset.path_data(parser)
argset.path_result(parser)
argset.position_size(parser)
_args = parser.parse_args()


class Args:
    watch = _args.watch
    episodes = _args.episodes
    save_anim = _args.save_anim
    cf_simenv = _args.cf_simenv
    path_data = _args.path_data
    path_result = _args.path_result
    position_size = _args.position_size


args = Args()


def env_test():
    if args.save_anim and args.watch != "plt":
        Color.print(
            "Ignore -s, --save-anim option: Use --watch=plt option to save videos", c=Color.coral
        )

    if args.save_anim and args.watch == "plt":
        checker.large_episodes(args.episodes)

    params_env = ParamsSimEnv(args.cf_simenv)
    print("params env:")
    print(params_env)

    T = params_env.max_episode_length // params_env.action_repeat

    env = ControlSuiteEnvWrap(**params_env.kwargs)

    print("observation size:", env.observation_size)
    print("action size:", env.action_size)
    print("action range:", env.action_range)

    if args.watch is None:
        path_data = Path(args.path_data)
        if path_data.exists():
            print(f'\n"{args.path_data}" directory will be rewritten.')
            if input("Do you want to continue? [y/n] ") != "y":
                print("Abort.")
                return
            shutil.rmtree(path_data)

        path_data.mkdir(parents=True, exist_ok=True)
        backup(args.cf_simenv, path_data, "params_env_bk.json5")

    if args.watch == "plt":

        def on_close(event):
            sys.exit()

        fig = plt.figure(figsize=figsize)
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
            self.init_LOG()

            self.step = 0
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
            mod = frame_cnt % T
            if mod == 0:
                self.init_LOG()

                env.reset()
                self.step = 0
                self.episode_cnt = frame_cnt // T + mod

            # ======================================================

            self.step += 1

            ### core ###
            action = env.sample_random_action()
            # action = env.zeros_action()
            observation, _, done, position = env.step(action)
            ###---------
            self.LOG_action.append(action.cpu().numpy())
            self.LOG_observation.append(observation.squeeze(0).cpu().numpy())
            self.LOG_delta.append(0.1)
            self.LOG_position.append(position)
            ############

            if args.watch == "render":
                env.render()

            elif args.watch == "plt":
                axes.clear()

                fig.suptitle(
                    f"episode: {self.episode_cnt+1}, step: {self.step:3d}",
                    fontname="monospace",
                )

                color_action = cmap(len(action), "prism")

                ax = axes.action
                ax.set_title("$\mathbf{u}_{t-1}$")
                ax.set_ylim(-1.2, 1.2)
                ax.set_aspect(0.7)
                ax.bar(range(len(action)), action, color=color_action, width=0.5)
                ax.set_xticks(range(len(action)))

                domain, task = params_env.env.split("-")

                if domain == "point_mass" and task == "easy":
                    ax.set_xticklabels(
                        ["$\mathbf{u}[0]$ : x (horizontal)", "$\mathbf{u}[1]$ : y (vertical)"]
                    )
                elif domain == "reacher":
                    ax.set_xticklabels(["$\mathbf{u}[0]$ : shoulder", "$\mathbf{u}[1]$ : wrist"])

                ax = axes.observation
                ax.set_title("$\mathbf{I}_t$")
                ax.imshow(mv.cnn2plt(obs2img(observation.squeeze(0).cpu())))

                wall = args.position_size
                # wall = 0.35
                ax = axes.position
                ax.set_title("Position")
                ax.set_xlim(-wall, wall)
                ax.set_ylim(-wall, wall)
                ax.set_aspect(1)
                ax.hlines(0, -wall, wall, color="black")
                ax.vlines(0, -wall, wall, color="black")
                ax.plot(position[0], position[1], marker="o", ms=10, color="orange")

                fig.tight_layout()

            if done:
                print(f"episode: {self.episode_cnt+1}, step: {self.step}")

                if args.watch is None:
                    episode_dir = Path(args.path_data, "episodes", f"{self.episode_cnt}")
                    episode_dir.mkdir(parents=True, exist_ok=True)
                    self.save_LOG(episode_dir)
                    Color.print("saved", c=Color.green)
                else:
                    Color.print("not saved", c=Color.coral)

    p = AnimPack()

    if args.watch == "plt":

        mpu.anim_mode(
            "save" if args.save_anim else "anim",
            fig,
            p.anim_func,
            T * args.episodes,
            interval=40,
            save_path=Path(args.path_result, f"{params_env.env}.mp4"),
        )

    else:
        for frame_cnt in range(T * args.episodes):
            p.anim_func(frame_cnt)


if __name__ == "__main__":
    env_test()
