import argparse
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec

import mypython.error as merror
import mypython.plotutil as mpu
import mypython.vision as mv
from mypython.plotutil import cmap
from simulation.env import obs2img
from tool import argset
from tool.dataloader import DataLoader

try:
    import tool._plot_config

    figsize = tool._plot_config.figsize_show_loss
except:
    figsize = None


parser = argparse.ArgumentParser(allow_abbrev=False)
argset.episodes(parser, required=False, default=10)
argset.path_data(parser)
argset.save_anim(parser)
argset.output(parser, help="Path of the video to be saved (Extension is .mp4, etc.)")
_args = parser.parse_args()


class Args:
    episodes = _args.episodes
    path_data = _args.path_data
    save_anim = _args.save_anim
    output = _args.output


args = Args()

if args.save_anim:
    assert args.output is not None


def main():
    merror.check_dir(args.path_data)
    max_episode = len([p for p in Path(args.path_data).glob("*") if p.is_dir()])

    dataloader = DataLoader(
        root=args.path_data,
        start=0,
        stop=max_episode,
        batch_size=args.episodes,
        dtype=torch.float32,
    )
    action, observation = next(dataloader)

    T = action.shape[0]
    dim_a = action.shape[-1]

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

            ax = axes.action
            ax.set_title("$\mathbf{u}_{t-1}$")
            l = 1.5
            ax.set_ylim(-l, l)
            a = action[self.t, self.episode_cnt]
            ax.bar(range(dim_a), a, color=cmap(dim_a, "prism"), width=0.5)
            ax.set_xticks(range(dim_a))

            # ax.set_xticklabels(
            #     ["$\mathbf{u}[0]$ : x (horizontal)", "$\mathbf{u}[1]$ : y (vertical)"]
            # )
            # ax.set_xticklabels(["$\mathbf{u}[0]$ : shoulder", "$\mathbf{u}[1]$ : wrist"])

            ax = axes.observation
            ax.set_title("$\mathbf{I}_t$")
            img = mv.cnn2plt(obs2img(observation[self.t, self.episode_cnt]))
            ax.imshow(img)

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
