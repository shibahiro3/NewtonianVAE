import sys

sys.path.append("../")

import argparse
import shutil
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import mypython.plot_config  # noqa: F401
import mypython.plotutil as mpu
import mypython.vision as mv
import numpy as np
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec

import tool.argset as argset
import tool.util
from env import ControlSuiteEnvWrap, obs2img
from tool.params import Params, ParamsSimEnv

parser = argparse.ArgumentParser(allow_abbrev=False)
argset.watch(parser)
argset.episodes(parser)
argset.save_anim(parser)
argset.cf(parser)
argset.cf_simenv(parser)
args = parser.parse_args()


def env_test():

    params = Params(args.cf)
    params_env = ParamsSimEnv(args.cf_simenv)

    env = ControlSuiteEnvWrap(**params_env.kwargs)

    print("observation size:", env.observation_size)
    print("action size:", env.action_size)
    print("action range:", env.action_range)

    if args.save_anim and args.watch != "plt":
        print("ignore --save-anim")

    if args.watch is None:
        path_data = Path(params.path.data)
        if path_data.exists():
            print(f'\nThe "{params.path.data}" directory will be rewritten.')
            if input("Do you want to continue? [y/n] ") != "y":
                print("Abort.")
                return
            shutil.rmtree(path_data)

        path_data.mkdir(parents=True, exist_ok=True)
        shutil.copy(args.cf_simenv, path_data)
        Path(path_data, Path(args.cf_simenv).name).chmod(0o444)  # read only

    if args.watch == "plt":

        def on_close(event):
            sys.exit()

        fig = plt.figure()
        fig.canvas.mpl_connect("close_event", on_close)
        gs = GridSpec(nrows=1, ncols=2)

        axes: Dict[str, Axes] = {}
        axes["action"] = fig.add_subplot(gs[0, 0])
        axes["observation"] = fig.add_subplot(gs[0, 1])

    class AnimPack:
        def __init__(self) -> None:
            self.LOG_action = []
            self.LOG_observation = []
            self.step = 0
            self.episode_cnt = 0

        def anim_func(self, frame_cnt):
            mod = frame_cnt % params.general.steps
            if mod == 0:
                self.LOG_action = []
                self.LOG_observation = []

                env.reset()
                self.step = 0
                self.episode_cnt = frame_cnt // params.general.steps + mod

            # ======================================================

            self.step += 1

            action = env.sample_random_action()
            observation, _, done, position = env.step(action)

            color_action = tool.util.cmap_plt(params.newtonian_vae.dim_x, "prism")

            self.LOG_action.append(action.cpu().numpy())
            self.LOG_observation.append(observation.squeeze(0).cpu().numpy())

            if args.watch == "render":
                env.render()

            elif args.watch == "plt":
                for ax in axes.values():
                    ax.clear()

                fig.suptitle(
                    f"episode: {self.episode_cnt+1}, step: {self.step:3d}",
                    fontname="monospace",
                )

                ax = axes["action"]
                ax.clear()
                ax.set_title("$\mathbf{u}_{t-1}$")
                ax.set_ylim(-1.2, 1.2)
                ax.bar(range(len(action)), action, color=color_action, width=0.5)
                ax.set_xticks(range(len(action)))

                domain, task = params_env.env.split("-")
                if domain == "point_mass" and task == "easy":
                    ax.set_xticklabels(
                        ["$\mathbf{u}[0]$ : x (horizontal)", "$\mathbf{u}[1]$ : y (vertical)"]
                    )
                elif domain == "reacher":
                    ax.set_xticklabels(["$\mathbf{u}[0]$ : shoulder", "$\mathbf{u}[1]$ : wrist"])

                ax = axes["observation"]
                ax.clear()
                ax.set_title("$\mathbf{I}_t$")
                ax.imshow(mv.cnn2plt(obs2img(observation.squeeze(0).cpu())))

                # fig.canvas.draw()
                # fig.canvas.flush_events()
                fig.tight_layout()

            if done:
                print(f"episode: {self.episode_cnt+1}, step: {self.step}")

                if args.watch is None:
                    episode_dir = Path(params.path.data, f"{self.episode_cnt}")
                    episode_dir.mkdir(parents=True, exist_ok=True)
                    np.save(Path(episode_dir, "action.npy"), self.LOG_action)
                    np.save(Path(episode_dir, "observation.npy"), self.LOG_observation)
                    print("saved")
                else:
                    print("not saved")

    p = AnimPack()

    if args.watch == "plt":

        mpu.anim_mode(
            "save" if args.save_anim else "anim",
            fig,
            p.anim_func,
            params.general.steps * args.episodes,
            interval=40,
            save_path=Path(params.path.result, f"{params_env.env}.mp4"),
        )

    else:
        for frame_cnt in range(params.general.steps * args.episodes):
            p.anim_func(frame_cnt)


if __name__ == "__main__":
    # print(args.watch)
    # print(args.save_anim)
    env_test()
