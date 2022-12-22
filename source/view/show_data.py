from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec

import mypython.error as merror
import mypython.plotutil as mpu
import mypython.vision as mv
import tool.plot_config
from mypython.ai.util import SequenceDataLoader
from simulation.env import obs2img

tool.plot_config.apply()
try:
    import tool._plot_config

    tool._plot_config.apply()
except:
    pass


def main(
    episodes: int,
    path_data: str,
    save_anim: bool,
    output: str,
):
    if save_anim:
        assert output is not None

    # ============================================================
    plt.rcParams.update(
        {
            "figure.figsize": (7.97, 3.44),
        }
    )

    fig = plt.figure()
    mpu.get_figsize(fig)

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
    # ============================================================

    merror.check_dir(path_data)
    max_episode = len([p for p in Path(path_data, "episodes").glob("*") if p.is_dir()])

    dataloader = SequenceDataLoader(
        root=Path(path_data, "episodes"),
        names=["action", "observation", "position"],
        start=0,
        stop=max_episode,
        batch_size=episodes,
        dtype=torch.float32,
    )
    action, observation, position = next(dataloader)

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

            dim_colors = mpu.cmap(dim_a, "prism")

            # ============================================================
            ax = axes.action
            ax.set_title(r"$\mathbf{u}_{t-1}$")
            ax.set_ylim(action.min() - 0.1, action.max() + 0.1)
            ax.bar(
                range(dim_a),
                action[self.t, self.episode_cnt],
                color=dim_colors,
                width=0.5,
            )
            # ax.set_xticks(range(dim_a))
            ax.tick_params(bottom=False, labelbottom=False)
            mpu.Axis_aspect_2d(ax, 1)

            # ============================================================
            ax = axes.observation
            ax.set_title(r"$\mathbf{I}_t$")
            ax.imshow(obs2img(observation[self.t, self.episode_cnt]))

            # ============================================================
            ax = axes.position
            ax.set_title(r"Position")
            ax.set_ylim(position.min() - 0.1, position.max() + 0.1)
            ax.bar(
                range(dim_a),
                position[self.t, self.episode_cnt],
                color=dim_colors,
                width=0.5,
            )
            # ax.set_xticks(range(dim_a))
            ax.tick_params(bottom=False, labelbottom=False)
            mpu.Axis_aspect_2d(ax, 1)

    p = AnimPack()
    mpu.anim_mode(
        "save" if save_anim else "anim",
        fig,
        p.anim_func,
        T * episodes,
        interval=40,
        save_path=output,
    )


if __name__ == "__main__":
    main()
