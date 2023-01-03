import dataclasses
import sys
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
from torch import Tensor

import controller.train as ct
import json5
import models.core
import models.pcontrol
import mypython.plotutil as mpu
import mypython.vision as mv
import tool.util
import view.plot_config
from models.cell import CellWrap
from models.core import NewtonianVAEFamily
from models.pcontrol import PControl
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
    config_ctrl: str,
    episodes: int,
    fix_xmap_size: float,
    save_anim: bool,
    steps: int,
    format: str,
):
    if save_anim:
        checker.large_episodes(episodes)

    torch.set_printoptions(precision=4, sci_mode=False)

    params_ctrl = paramsmanager.Params(config_ctrl)
    dtype, device = tool.util.dtype_device(
        dtype=params_ctrl.eval.dtype,
        device=params_ctrl.eval.device,
    )

    p_pctrl, manage_dir, weight_path, saved_params_ctrl = tool.util.load(
        root=params_ctrl.path.saves_dir,
        model_place=models.pcontrol,
    )
    p_pctrl: PControl
    p_pctrl.type(dtype)
    p_pctrl.to(device)
    p_pctrl.eval()

    model, _ = tool.util.load_direct(
        weight_path=saved_params_ctrl.path.used_nvae_weight, model_place=models.core
    )
    model: NewtonianVAEFamily
    model.type(dtype)
    model.to(device)
    model.eval()

    path_result = tool.util.priority(
        params_ctrl.path.results_dir, saved_params_ctrl.path.results_dir
    )
    env = ControlSuiteEnvWrap(**params_ctrl.raw_["ControlSuiteEnvWrap"])

    if steps is not None:
        max_time_length = steps
    else:
        max_time_length = params_ctrl.train.max_time_length

    del params_ctrl
    del saved_params_ctrl

    print("=== PControl params ===")
    Color.print("x^goal_n:")
    print(p_pctrl.x_goal_n.detach().cpu())
    Color.print("K_n")
    print(p_pctrl.K_n.detach().cpu())

    record = calculate(
        p_pctrl=p_pctrl, model=model, env=env, episodes=episodes, time_length=max_time_length
    )

    all_steps = max_time_length * episodes

    p = draw(
        record=record,
        model=model,
        env=env,
        p_pctrl=p_pctrl,
        max_time_length=max_time_length,
        all_steps=all_steps,
        fix_xmap_size=fix_xmap_size,
    )

    save_path = Path(path_result, f"{manage_dir.stem}_W{weight_path.stem}_control.{format}")
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
    pi: np.ndarray


def calculate(
    p_pctrl: PControl,
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
            x_t_ = ct.preprocess_x(x_t)
            action_ = p_pctrl.step(x_t_)
            action = ct.postprocess_u(action_)

            observation, _, done, position = env.step(action)
            observation = observation.to(dtype=dtype).to(device=device)

            pi, mu, sigma = p_pctrl.dist_parameters()

            record.append(
                action=to_np(action_prev.squeeze(0)),
                observation=to_np(observation.squeeze(0)),
                reconstruction=to_np(I_t_dec.squeeze(0)),
                x=to_np(x_t.squeeze(0)),
                position=position,
                pi=to_np(pi.squeeze(0)),
            )

            Prompt.print_one_line(
                f"Trial: {episode+1:2d}/{episodes} | Step: {t+1:4d}/{time_length} "
            )

    print("\nDone")
    record.to_whole_np(show_shape=True)

    return record


def draw(
    record: Pack,
    p_pctrl: PControl,
    model: NewtonianVAEFamily,
    env: ControlSuiteEnvWrap,
    max_time_length: int,
    all_steps: int,
    fix_xmap_size: float,
):
    dim_colors = mpu.cmap(model.cell.dim_x, "prism")
    label = Label(env.domain)

    plt.rcParams.update(
        {
            "figure.figsize": (12.76, 8.39),
            "figure.subplot.left": 0.07,
            "figure.subplot.right": 0.98,
            "figure.subplot.bottom": 0.1,
            "figure.subplot.top": 0.9,
            "figure.subplot.hspace": 0,
            "figure.subplot.wspace": 0,
        }
    )

    fig = plt.figure()
    mpu.get_figsize(fig)

    class Ax:
        def __init__(self) -> None:
            r = Seq2(2, (6, 1), lazy=True)
            c1 = Seq2(3, (4, 1))
            c2 = Seq2(4, (2, 1))
            length = Seq2.share_length(c1, c2)
            gs = GridSpec(nrows=r.length, ncols=length)

            self.action = fig.add_subplot(gs[r.a : r.b, c1.a : c1.b])
            self.observation = fig.add_subplot(gs[r.a : r.b, c1.a : c1.b])
            self.obs_dec = fig.add_subplot(gs[r.a : r.b, c1.a : c1.b])
            r.update()
            self.pi = fig.add_subplot(gs[r.a : r.b, c2.a : c2.b])
            self.x_map = fig.add_subplot(gs[r.a : r.b, c2.a : c2.b])
            self.x = fig.add_subplot(gs[r.a : r.b, c2.a : c2.b])
            self.y = fig.add_subplot(gs[r.a : r.b, c2.a : c2.b])

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
            ax = axes.pi
            ax.set_title(r"${\pi}_{k}(\mathbf{x})$")
            ax.bar(
                range(record.pi.shape[-1]),
                record.pi[self.episode, self.t],
                width=0.5,
            )
            ax.set_ylim(0, 1.1)
            mpu.Axis_aspect_2d(ax, 1)
            # ax.tick_params(labelbottom=False)
            ax.set_xticks(range(record.pi.shape[-1]))
            ax.set_xticklabels([" "] * record.pi.shape[-1])

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
                p_pctrl.x_goal_n.squeeze(0)[:, 0].cpu(),
                p_pctrl.x_goal_n.squeeze(0)[:, 1].cpu(),
                marker="o",
                s=20,
                color="purple",
                label=r"$\mathbf{x}_{goal}$",
            )
            # ax.legend(loc="lower left")
            ax.legend()

            min_ = min(p_pctrl.x_goal_n.squeeze(0)[:, 0].cpu().min(), record.x[..., 0].min())
            max_ = max(p_pctrl.x_goal_n.squeeze(0)[:, 0].cpu().max(), record.x[..., 0].max())
            margin_ = (max_ - min_) * 0.1
            ax.set_xlim(min_ - margin_, max_ + margin_)

            min_ = min(p_pctrl.x_goal_n.squeeze(0)[:, 1].cpu().min(), record.x[..., 1].min())
            max_ = max(p_pctrl.x_goal_n.squeeze(0)[:, 1].cpu().max(), record.x[..., 1].max())
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
