import shutil
import sys
from pathlib import Path

import classopt
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec

import mypython.error as merror
import mypython.plotutil as mpu
import mypython.vision as mv
import tool.plot_config
from mypython.ai.util import SequenceDataLoader
from mypython.plotutil import cmap
from simulation.env import obs2img
from tool import argset

tool.plot_config.apply()
try:
    import tool._plot_config

    tool._plot_config.apply()
except:
    pass


@classopt.classopt(default_long=True, default_short=False)
class Args:
    episodes: int = classopt.config(**argset.descr_episodes, required=True)
    path_data: str = classopt.config(**argset.descr_path_data, required=False)
    save_anim: bool = classopt.config()
    output: str = classopt.config(
        metavar="ENV", help="Path of the video to be saved (Extension is .mp4, etc.)"
    )


args = Args.from_args()  # pylint: disable=E1101

if args.save_anim:
    assert args.output is not None


def main():
    # ============================================================
    fig = plt.figure()
    mpu.get_figsize(fig)

    class Ax:
        def __init__(self) -> None:
            gs = GridSpec(nrows=1, ncols=2)
            self.action = fig.add_subplot(gs[0, 0])
            self.observation = fig.add_subplot(gs[0, 1])

        def clear(self):
            for ax in self.__dict__.values():
                ax.clear()

    axes = Ax()
    # ============================================================

    merror.check_dir(args.path_data)
    max_episode = len([p for p in Path(args.path_data).glob("*") if p.is_dir()])

    dataloader = SequenceDataLoader(
        root=args.path_data,
        names=["action", "observation"],
        start=0,
        stop=max_episode,
        batch_size=args.episodes,
        dtype=torch.float32,
    )
    action, observation = next(dataloader)

    T = action.shape[0]
    dim_a = action.shape[-1]

    class AnimPack:
        def __init__(self) -> None:
            self.t = 0
            self.episode_cnt = 0

        def anim_func(self, frame_cnt):
            axes.clear()

            mod = frame_cnt % T
            if mod == 0:
                self.t = 0
                self.episode_cnt = frame_cnt // T + mod
            else:
                self.t += 1

            fig.suptitle(
                f"Sampled episode: {self.episode_cnt+1}, t = {self.t:3d}, T = {T}",
                fontname="monospace",
            )

            # ============================================================
            ax = axes.action
            ax.set_title(r"$\mathbf{u}_{t-1}$")
            l = 1.5
            ax.set_ylim(-l, l)
            a = action[self.t, self.episode_cnt]
            ax.bar(range(dim_a), a, color=cmap(dim_a, "prism"), width=0.5)
            ax.set_xticks(range(dim_a))

            # ============================================================
            ax = axes.observation
            ax.set_title(r"$\mathbf{I}_t$")
            ax.imshow(obs2img(observation[self.t, self.episode_cnt]))

    p = AnimPack()
    mpu.anim_mode(
        "save" if args.save_anim else "anim",
        fig,
        p.anim_func,
        T * args.episodes,
        interval=40,
        save_path=args.output,
    )


if __name__ == "__main__":
    main()
