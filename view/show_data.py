import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec

import mypython.error as merror
import mypython.plot_config  # noqa: F401
import mypython.plotutil as mpu
import mypython.vision as mv
import tool.argset as argset
from simulation.env import obs2img
from tool.dataloader import GetBatchData
from tool.params import Params, ParamsSimEnv
from tool.util import cmap_plt

parser = argparse.ArgumentParser(allow_abbrev=False)
argset.episodes(parser, required=False, default=10)
argset.path_data(parser)
argset.save_anim(parser)
parser.add_argument(
    "-o",
    "--output",
    metavar="PATH",
    type=str,
    help="Path of the video to be saved (Extension is .mp4, etc.)",
)
args = parser.parse_args()


if args.save_anim:
    assert args.output is not None


def main():
    merror.check_dir(args.path_data)
    max_episode = len([p for p in Path(args.path_data).glob("*") if p.is_dir()])

    BatchData = GetBatchData(
        args.path_data,
        0,
        max_episode,
        args.episodes,
        dtype=torch.float32,
    )
    action, observation = next(BatchData)
    T = action.shape[0]
    dim_a = action.shape[-1]

    fig = plt.figure()
    gs = GridSpec(nrows=1, ncols=2)

    axes: Dict[str, Axes] = {}
    axes["action"] = fig.add_subplot(gs[0, 0])
    axes["observation"] = fig.add_subplot(gs[0, 1])

    class AnimPack:
        def __init__(self) -> None:
            self.t = 0
            self.episode_cnt = 0

        def anim_func(self, frame_cnt):
            mod = frame_cnt % T
            if mod == 0:
                self.t = 0
                self.episode_cnt = frame_cnt // T + mod
            else:
                self.t += 1

            for ax in axes.values():
                ax.clear()

            fig.suptitle(
                f"episode (random): {self.episode_cnt+1}, t = {self.t:3d}, T = {T}",
                fontname="monospace",
            )

            ax = axes["action"]
            ax.clear()
            ax.set_title("$\mathbf{u}_{t-1}$")
            l = 1.5
            ax.set_ylim(-l, l)
            a = action[self.t, self.episode_cnt]
            ax.bar(range(dim_a), a, color=cmap_plt(dim_a, "prism"), width=0.5)
            ax.set_xticks(range(dim_a))

            # ax.set_xticklabels(
            #     ["$\mathbf{u}[0]$ : x (horizontal)", "$\mathbf{u}[1]$ : y (vertical)"]
            # )
            # ax.set_xticklabels(["$\mathbf{u}[0]$ : shoulder", "$\mathbf{u}[1]$ : wrist"])

            ax = axes["observation"]
            ax.clear()
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
