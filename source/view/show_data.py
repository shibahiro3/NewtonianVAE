from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec

import mypython.error as merror
import mypython.plotutil as mpu
import mypython.vision as mv
import view.plot_config
from mypython.ai.util import SequenceDataLoader
from mypython.terminal import Color, Prompt
from simulation.env import obs2img
from tool import paramsmanager

view.plot_config.apply()
try:
    import view._plot_config

    view._plot_config.apply()
except:
    pass


def main(
    config: str,
    episodes: int,
    save_anim: bool,
    format: str,
):
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

    params = paramsmanager.Params(config)

    merror.check_dir(params.path.data_dir)

    dataloader = SequenceDataLoader(
        root=Path(params.path.data_dir, "episodes"),
        names=["action", "observation", "position"],
        start=params.train.data_start,
        stop=params.train.data_stop,
        batch_size=episodes,
        dtype=torch.float32,
        show_selected_index=True,
    )
    action, observation, position = next(dataloader)

    T = action.shape[0]
    dim_a = action.shape[-1]
    all_steps = T * episodes

    class AnimPack:
        def __init__(self) -> None:
            self.t = 0
            self.episode_cnt = 0

        def anim_func(self, frame_cnt):
            axes.clear()

            Prompt.print_one_line(
                f"{frame_cnt+1:5d} / {all_steps} ({(frame_cnt+1)*100/all_steps:.1f} %) "
            )

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
            ax.set_xlabel(f"{observation.shape[-1]} px")
            ax.set_ylabel(f"{observation.shape[-2]} px")
            ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)

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
        all_steps,
        interval=40,
        save_path=Path(params.path.data_dir, f"data.{format}"),
    )
