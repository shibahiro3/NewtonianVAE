"""
It doesn't train the action model (policy model).
"""

import sys
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from torch import Tensor

import models.core
import mypython.plotutil as mpu
import mypython.plt_layout as pltl
import mypython.vision as mv
import tool.util
import view.plot_config
from models.controller import PurePControl
from models.core import NewtonianVAEBase
from mypython import rdict
from mypython.ai.util import to_numpy
from mypython.pyutil import add_version
from mypython.terminal import Color, Prompt
from simulation.env import ControlSuiteEnvWrap, img2obs, obs2img
from tool import checker, paramsmanager
from tool.util import RecoderBase
from view.label import Label


def main(
    config: str,
    episodes: int,
    fix_xmap_size: float,
    save_anim: bool,
    goal_img: str,
    alpha: float,
    steps: int,
    format: str,
):
    if save_anim:
        checker.large_episodes(episodes)

    torch.set_printoptions(precision=4, sci_mode=False)

    params = paramsmanager.Params(config)
    dtype, device = tool.util.dtype_device(
        dtype=params.train.dtype,
        device=params.train.device,
    )

    model, manage_dir, weight_path, saved_params = tool.util.load(
        root=params.path.saves_dir,
        model_place=models.core,
    )
    model: NewtonianVAEBase
    model.type(dtype)
    model.to(device)
    model.eval()

    path_result = params.path.results_dir
    env = ControlSuiteEnvWrap(**params.raw["ControlSuiteEnvWrap"])

    if steps is not None:
        max_time_length = steps
    else:
        max_time_length = saved_params.train.max_time_length

    del params
    del saved_params

    # Igoal = mv.cv2cnn(np.load(goal_img)).unsqueeze_(0).to(device)
    # ctrl = PurePControl(img2obs(Igoal), alpha, model.cell)

    record = calculate(model=model, env=env, P=alpha, episodes=episodes, T=max_time_length)

    all_steps = max_time_length * episodes

    p = draw(
        record=record,
        fix_xmap_size=fix_xmap_size,
        all_steps=all_steps,
        max_time_length=max_time_length,
        P_gain=alpha,
    )
    save_path = Path(
        path_result, f"{manage_dir.stem}", f"E{weight_path.stem}_control_pure.{format}"
    )
    save_path = add_version(save_path)
    mpu.anim_mode(
        "save" if save_anim else "anim",
        p.fig,
        p.anim_func,
        all_steps,
        interval=40,
        freeze_cnt=-1,
        save_path=save_path,
    )

    print()


# @initialize.all_with([])
# class Pack(RecoderBase):
#     action: np.ndarray
#     observation: np.ndarray
#     reconstruction: np.ndarray
#     x: np.ndarray
#     position: np.ndarray


def calculate(
    *,
    model: NewtonianVAEBase,
    P: float,
    env: ControlSuiteEnvWrap,
    episodes: int,
    T: int,
) -> List[Dict[str, np.ndarray]]:
    """
    Returns:
        "action" :   [0,          u_0,     u_1,     u_2,  ...]  (u_0 = 0 vector)
        "camera"
          "top"  :   [I_goal,     I_1,     I_2,     ...]  (observation)
          "self" ...
          ...
        "x" :        [x_goal,     x_1,     x_2,     ...]
        "recon" :    [I_rec_goal, I_rec_1, I_rec_2, ...]
        "position" : [...]

            (1 + T) x episodes
    """

    torch.set_grad_enabled(False)

    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    model.eval()

    record = []

    print("Calculating...")
    for episode in range(episodes):
        record_one = {}

        zero = torch.zeros(size=(1, model.dim_x), device=device, dtype=dtype)

        o_target = env.reset()
        I_goal = o_target["camera"]
        position = o_target["position"]
        I_goal = {name: I_goal[name] for name in model.camera_names}
        rdict.to_torch_(I_goal, dtype=dtype, device=device)
        # rdict.show(I_goal, "I_goal")

        x_goal = model.encode([e.unsqueeze(0) for e in I_goal.values()])
        I_rec_goal = model.decode(x_goal)

        rdict.append_a_to_b(
            {
                "action": zero.squeeze(0),  # dummy
                "x": x_goal.squeeze(0),
                "camera": I_goal,
                "recon": rdict.apply(I_rec_goal, lambda x: x.squeeze(0)),
                "position": position,
            },
            record_one,
        )

        u_tn1 = zero
        env.reset()
        for t in range(T):
            o_t, done = env.step(u_tn1)
            I_t = o_t["camera"]
            position = o_t["position"]
            I_t = {name: I_t[name] for name in model.camera_names}
            rdict.to_torch_(I_t, dtype=dtype, device=device)

            x_t = model.encode([e.unsqueeze(0) for e in I_t.values()])
            I_rec_t = model.decode(x_t)

            rdict.append_a_to_b(
                {
                    "action": u_tn1.squeeze(0),
                    "x": x_t.squeeze(0),
                    "camera": I_t,
                    "recon": rdict.apply(I_rec_t, lambda x: x.squeeze(0)),
                    "position": position,
                },
                record_one,
            )

            u_t = P * (x_goal - x_t)
            u_tn1 = u_t

            Prompt.print_one_line(f"Trial: {episode+1:2d}/{episodes} | Step: {t+1:4d}/{T} ")

        rdict.to_numpy_(record_one)
        record.append(record_one)

    print("\nDone")

    # rdict.show(record[0], "record 0")
    # rdict.show(record[2], "record 2")
    return record


def draw(
    record,
    # env: ControlSuiteEnvWrap,
    # Igoal: Tensor,
    max_time_length: int,
    all_steps: int,
    fix_xmap_size: float,
    P_gain: float,
):
    dim_x = record[0]["action"].shape[-1]
    camera_names = record[0]["camera"].keys()

    colors_action = mpu.cmap(dim_x, "prism")
    colors_latent = mpu.cmap(dim_x, "rainbow")
    # label = Label(env.domain)

    # ============================================================
    plt.rcParams.update({"axes.titlesize": 12})

    fig = plt.figure(figsize=(8.77, 8.39))
    mpu.get_figsize(fig)
    # fig.subplots_adjust(top=0.7)

    class Ax:
        def __init__(self) -> None:
            fig.subplots_adjust(top=0.9, bottom=0.05)

            n_img = len(camera_names)

            self.action = pltl.Plotter()
            self.Igoals = [pltl.Plotter(clearable=False) for _ in range(n_img)]
            self.observations = [pltl.Plotter() for _ in range(n_img)]
            self.recon = [pltl.Plotter() for _ in range(n_img)]
            self.position = pltl.Plotter(flex=4)
            self.position_bar = pltl.Plotter()
            self.x = pltl.Plotter(flex=4)
            self.x_bar = pltl.Plotter()

            self.layout = pltl.Column(
                [
                    pltl.Row(
                        [
                            self.action,
                            pltl.Column(
                                [
                                    pltl.Row(self.Igoals),
                                    pltl.Row(self.observations),
                                    pltl.Row(self.recon),
                                ],
                                space=0.3,
                                flex=n_img,
                            ),
                        ],
                        # flex=2,
                    ),
                    pltl.Column(
                        [
                            pltl.Row([self.x, self.x_bar]),
                            pltl.Row([self.position, self.position_bar]),
                        ],
                        space=0.5,
                    ),
                ],
            )

            pltl.compile(fig, self.layout)

        def clear(self):
            pltl.clear(self.layout)

    axes = Ax()

    T = record[0]["action"].shape[0] - 1

    class AnimPack:
        def __init__(self) -> None:
            self.fig = fig

        def anim_func(self, frame_cnt):
            axes.clear()

            Prompt.print_one_line(
                f"{frame_cnt+1:5d} / {all_steps} ({(frame_cnt+1)*100/all_steps:.1f} %) "
            )

            mod = frame_cnt % max_time_length
            if mod == 0:
                self.t = -1
                self.episode = frame_cnt // max_time_length + mod

                self.action_min = record[self.episode]["action"].min()
                self.action_max = record[self.episode]["action"].max()

                self.position_min = record[self.episode]["position"].min()
                self.position_max = record[self.episode]["position"].max()

                self.x_min = record[self.episode]["x"].min()
                self.x_max = record[self.episode]["x"].max()

                pltl.clear(axes.Igoals)
                for i, k in enumerate(camera_names):
                    ax = axes.Igoals[i].ax
                    ax.set_title(f"Target Image ({k})")
                    ax.imshow(obs2img(record[self.episode]["camera"][k][0]))
                    ax.set_axis_off()

            self.t += 1
            # print(self.episode, self.t)

            # ============================================================

            fig.suptitle(
                f"Trial: {self.episode+1}, Step: {self.t:3d}",
                fontname="monospace",
            )

            # ============================================================
            ax = axes.action.ax
            N = dim_x
            R = range(1, N + 1)
            ax.set_title(r"$\mathbf{u}_{t-1}$ (Generated)")
            ax.bar(
                R,
                record[self.episode]["action"][self.t],
                color=colors_action,
                width=0.5,
            )
            pad = (self.action_max - self.action_min) * 0.1
            ax.set_ylim(self.action_min - pad, self.action_max + pad)
            ax.set_xticks(R)
            ax.set_xticklabels([str(s) for s in R])
            mpu.Axis_aspect_2d(ax, 1)

            # ============================================================
            for i, k in enumerate(camera_names):
                ax = axes.observations[i].ax
                ax.set_title(r"$\mathbf{I}_t$ " f"({k}, From environment)")
                ax.imshow(obs2img(record[self.episode]["camera"][k][1 + self.t]))
                ax.set_axis_off()

            # ============================================================
            for i, k in enumerate(camera_names):
                ax = axes.recon[i].ax
                ax.set_title(r"$\mathbf{I}_t$ " f"({k}, Reconstructed)")
                ax.imshow(obs2img(record[self.episode]["recon"][k][1 + self.t]))
                ax.set_axis_off()

            # ============================================================
            ax = axes.position.ax
            N = dim_x
            ax.set_title("Physical position" r"$_{1:t}$")
            ax.set_xlim(0, T)
            pad = (self.position_max - self.position_min) * 0.1
            ax.set_ylim(self.position_min - pad, self.position_max + pad)
            # ax.yaxis.set_major_formatter(FormatStrFormatter("%2.2f"))
            for i in range(N):
                ax.plot(
                    T,
                    record[self.episode]["position"][0, i],
                    color=colors_latent[i],
                    marker="o",
                    ms=10,
                    mec=None,
                )
            for i in range(N):
                ax.plot(
                    # list(range(self.t)),
                    record[self.episode]["position"][1 : 1 + self.t + 1, i],
                    color=colors_latent[i],
                    lw=1,
                )

            # ============================================================
            ax = axes.position_bar.ax
            N = dim_x
            R = range(1, N + 1)
            ax.set_title("Physical position" r"$_t$ ")

            pad = (self.position_max - self.position_min) * 0.1
            ax.set_ylim(self.position_min - pad, self.position_max + pad)
            # ax.yaxis.set_major_formatter(FormatStrFormatter("%2.2f"))
            ax.bar(
                R,
                record[self.episode]["position"][1 + self.t],
                color=colors_latent,
            )
            ax.set_xticks(R)
            ax.set_xticklabels([str(s) for s in R])
            ax.tick_params(left=False, labelleft=False)
            # ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
            mpu.Axis_aspect_2d(ax, 1)

            # ============================================================
            ax = axes.x.ax
            N = dim_x
            ax.set_title(r"$\mathbf{x}_{1:t}$" " (Latent)")
            ax.set_xlim(0, T)
            pad = (self.x_max - self.x_min) * 0.1
            ax.set_ylim(self.x_min - pad, self.x_max + pad)
            # ax.yaxis.set_major_formatter(FormatStrFormatter("%2.2f"))
            for i in range(N):
                ax.plot(
                    T,
                    record[self.episode]["x"][0, i],
                    color=colors_latent[i],
                    marker="o",
                    ms=10,
                    mec=None,
                )
            for i in range(N):
                ax.plot(
                    # list(range(self.t)),
                    record[self.episode]["x"][1 : 1 + self.t + 1, i],
                    color=colors_latent[i],
                    lw=1,
                )

            # ============================================================
            ax = axes.x_bar.ax
            N = dim_x
            R = range(1, N + 1)
            ax.set_title(r"$\mathbf{x}_t$")
            pad = (self.x_max - self.x_min) * 0.1
            ax.set_ylim(self.x_min - pad, self.x_max + pad)
            # ax.yaxis.set_major_formatter(FormatStrFormatter("%2.2f"))
            ax.bar(
                R,
                record[self.episode]["x"][1 + self.t],
                color=colors_latent,
            )
            ax.set_xticks(R)
            ax.set_xticklabels([str(s) for s in R])
            ax.tick_params(left=False, labelleft=False)
            # ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
            mpu.Axis_aspect_2d(ax, 1)

    return AnimPack()
