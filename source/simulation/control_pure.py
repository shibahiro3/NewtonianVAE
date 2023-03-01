"""
It doesn't train the action model (policy model).
"""

import sys
from pathlib import Path
from pprint import pprint
from typing import Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
from torch import Tensor

import models.core
import mypython.plotutil as mpu
import mypython.vision as mv
import tool.util
import view.plot_config
from models.cell import CellWrap
from models.core import NewtonianVAEFamily
from models.pcontrol import PurePControl
from mypython.ai.util import to_np
from mypython.pyutil import Seq, Seq2, add_version, initialize
from mypython.terminal import Color, Prompt
from simulation.env import ControlSuiteEnvWrap, img2obs, obs2img
from tool import checker, paramsmanager
from tool.util import RecoderBase
from view.label import Label

view.plot_config.apply()
try:
    import view._plot_config

    view._plot_config.apply()
except:
    pass


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
    model: NewtonianVAEFamily
    model.type(dtype)
    model.to(device)
    model.eval()

    path_result = tool.util.priority(params.path.results_dir, saved_params.path.results_dir)
    env = ControlSuiteEnvWrap(**params.raw["ControlSuiteEnvWrap"])

    if steps is not None:
        max_time_length = steps
    else:
        max_time_length = saved_params.train.max_time_length

    del params
    del saved_params

    Igoal = mv.cv2cnn(np.load(goal_img)).unsqueeze_(0).to(device)
    ctrl = PurePControl(img2obs(Igoal), alpha, model.cell)
    record = calculate(
        ctrl=ctrl, model=model, env=env, episodes=episodes, time_length=max_time_length
    )

    all_steps = max_time_length * episodes

    p = draw(
        record=record,
        model=model,
        ctrl=ctrl,
        env=env,
        Igoal=Igoal,
        fix_xmap_size=fix_xmap_size,
        all_steps=all_steps,
        max_time_length=max_time_length,
    )
    save_path = Path(path_result, f"{manage_dir.stem}_W{weight_path.stem}_control_pure.{format}")
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


@initialize.all_with([])
class Pack(RecoderBase):
    action: np.ndarray
    observation: np.ndarray
    reconstruction: np.ndarray
    x: np.ndarray
    position: np.ndarray


def calculate(
    ctrl: PurePControl,
    model: NewtonianVAEFamily,
    env: ControlSuiteEnvWrap,
    episodes: int,
    time_length: int,
):
    torch.set_grad_enabled(False)

    record = Pack()

    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device

    print("Calculating...")
    for episode in range(episodes):
        record.add_list()

        cell_wrap = CellWrap(cell=model.cell)

        action = torch.zeros(size=(1, model.cell.dim_x), device=device, dtype=dtype)
        observation = env.reset().to(dtype=dtype).to(device=device)

        for t in range(time_length):
            action_prev = action  # for recording

            x_t, I_t_dec = cell_wrap.step(action=action_prev, observation=observation)
            action = ctrl.step(x_t)

            observation, _, done, position = env.step(action)
            observation = observation.to(dtype=dtype).to(device=device)

            record.append(
                action=to_np(action_prev.squeeze(0)),
                observation=to_np(observation.squeeze(0)),
                reconstruction=to_np(I_t_dec.squeeze(0)),
                x=to_np(x_t.squeeze(0)),
                position=position,
            )

            Prompt.print_one_line(
                f"Trial: {episode+1:2d}/{episodes} | Step: {t+1:4d}/{time_length} "
            )

    print("\nDone")
    record.to_whole_np(show_shape=True)

    return record


def draw(
    record: Pack,
    model: NewtonianVAEFamily,
    ctrl: PurePControl,
    env: ControlSuiteEnvWrap,
    Igoal: Tensor,
    max_time_length: int,
    all_steps: int,
    fix_xmap_size: float,
):
    dim_colors = mpu.cmap(model.cell.dim_x, "prism")
    label = Label(env.domain)

    plt.rcParams.update(
        {
            "figure.figsize": (12.76, 8.39),
            "figure.subplot.left": 0.05,
            "figure.subplot.right": 0.95,
            "figure.subplot.bottom": 0.1,
            "figure.subplot.top": 1,
            "figure.subplot.hspace": 0.3,
            "figure.subplot.wspace": 0.3,
        }
    )

    fig = plt.figure()
    mpu.get_figsize(fig)

    class Ax:
        def __init__(self) -> None:
            gs = GridSpec(nrows=5, ncols=12)
            up = 3

            v1 = Seq(step=3, start_now=True)
            v2 = Seq(step=4, start_now=True)

            self.action = fig.add_subplot(gs[:up, v1.now : v1.next()])
            self.Igoal = fig.add_subplot(gs[:up, v1.now : v1.next()])
            self.observation = fig.add_subplot(gs[:up, v1.now : v1.next()])
            self.obs_dec = fig.add_subplot(gs[:up, v1.now : v1.next()])

            self.x_map = fig.add_subplot(gs[up:5, v2.now : v2.next()])
            self.x = fig.add_subplot(gs[up:5, v2.now : v2.next()])
            self.y = fig.add_subplot(gs[up:5, v2.now : v2.next()])

        def clear(self):
            for ax in self.__dict__.values():
                ax.clear()

    axes = Ax()

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

            self.t += 1
            # print(self.episode, self.t)

            # ============================================================

            fig.suptitle(
                f"Trial: {self.episode+1}, Step: {self.t:3d}",
                fontname="monospace",
            )

            # ============================================================
            ax = axes.action
            ax.set_title(r"$\mathbf{u}_{t-1}$ (Generated)")
            ax.bar(
                range(model.cell.dim_x),
                record.action[self.episode, self.t],
                color=dim_colors,
                width=0.5,
            )

            min_ = record.action.min()
            max_ = record.action.max()
            margin_ = (max_ - min_) * 0.1
            ax.set_ylim(min_ - margin_, max_ + margin_)

            # ax.tick_params(bottom=False, labelbottom=False)
            mpu.Axis_aspect_2d(ax, 1)
            ax.set_xticks(range(model.cell.dim_x))
            if env.domain == "reacher2d":
                ax.set_xlabel("Torque")
                ax.set_xticklabels([r"$\mathbf{u}[1]$ (shoulder)", r"$\mathbf{u}[2]$ (wrist)"])
            elif env.domain == "point_mass" and env.task == "easy":
                ax.set_xticklabels([r"$\mathbf{u}[1]$ (x)", r"$\mathbf{u}[2]$ (y)"])

            # ============================================================
            ax = axes.observation
            ax.set_title(r"$\mathbf{I}_t$ (From environment)")
            ax.imshow(obs2img(record.observation[self.episode, self.t]))
            ax.set_axis_off()

            # ============================================================
            ax = axes.obs_dec
            ax.set_title(r"$\mathbf{I}_t$ (Reconstructed)")
            ax.imshow(obs2img(record.reconstruction[self.episode, self.t]))
            ax.set_axis_off()

            # ============================================================
            ax = axes.Igoal
            ax.set_title(r"$\mathbf{I}_{goal}$")
            ax.imshow(mv.cnn2plt(Igoal.detach().squeeze(0).cpu()))
            ax.set_axis_off()

            # ============================================================
            ax = axes.x_map
            ax.set_title(r"$\mathbf{x}_{1:t}$")
            ax.plot(
                record.x[self.episode, : self.t + 1, 0],
                record.x[self.episode, : self.t + 1, 1],
                marker="o",
                ms=2,
            )
            ax.scatter(
                record.x[self.episode, self.t, 0],
                record.x[self.episode, self.t, 1],
                marker="o",
                s=20,
                color="red",
                label=r"$\mathbf{x}_{t}$",
            )
            ax.scatter(
                ctrl.x_goal[..., 0].cpu(),
                ctrl.x_goal[..., 1].cpu(),
                marker="o",
                s=20,
                color="purple",
                label=r"$\mathbf{x}_{goal}$",
            )
            # ax.legend(loc="lower left")
            ax.legend()

            min_ = record.x[..., 0].min()
            max_ = record.x[..., 0].max()
            margin_ = (max_ - min_) * 0.1
            ax.set_xlim(min_ - margin_, max_ + margin_)

            min_ = record.x[..., 1].min()
            max_ = record.x[..., 1].max()
            margin_ = (max_ - min_) * 0.1
            ax.set_ylim(min_ - margin_, max_ + margin_)

            label.set_axes_L0L1(ax, fix_xmap_size)

            # ============================================================
            ax = axes.x
            idx = 0
            ax.plot(
                record.position[self.episode, : self.t + 1, idx],
                record.x[self.episode, : self.t + 1, idx],
                marker="o",
                ms=2,
            )
            ax.scatter(
                record.position[self.episode, self.t, idx],
                record.x[self.episode, self.t, idx],
                marker="o",
                s=20,
                color="red",
                label=r"$\mathbf{x}_{t}$",
            )
            min_ = record.x[..., idx].min()
            max_ = record.x[..., idx].max()
            margin_ = (max_ - min_) * 0.1
            ax.set_ylim(min_ - margin_, max_ + margin_)

            label.set_axes_P0L0(ax, fix_xmap_size)

            # ============================================================
            ax = axes.y
            idx = 1
            ax.plot(
                record.position[self.episode, : self.t + 1, idx],
                record.x[self.episode, : self.t + 1, idx],
                marker="o",
                ms=2,
            )
            ax.scatter(
                record.position[self.episode, self.t, idx],
                record.x[self.episode, self.t, idx],
                marker="o",
                s=20,
                color="red",
                label=r"$\mathbf{x}_{t}$",
            )
            min_ = record.x[..., idx].min()
            max_ = record.x[..., idx].max()
            margin_ = (max_ - min_) * 0.1
            ax.set_ylim(min_ - margin_, max_ + margin_)

            label.set_axes_P1L1(ax, fix_xmap_size)

    return AnimPack()
