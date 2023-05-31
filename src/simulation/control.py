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
import models.controller
import models.core
import mypython.ai.torchprob as tp
import mypython.plotutil as mpu
import mypython.vision as mv
import tool.util
import view.plot_config
from models.controller import PControl
from models.core import NewtonianVAEBase
from mypython.ai.util import to_numpy
from mypython.pyutil import add_version, initialize
from mypython.terminal import Color, Prompt
from simulation.env import ControlSuiteEnvWrap, img2obs, obs2img
from tool import checker, paramsmanager
from tool.util import RecoderBase
from view.label import Label


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
        dtype=params_ctrl.test.dtype,
        device=params_ctrl.test.device,
    )

    p_pctrl, manage_dir, weight_path, saved_params_ctrl = tool.util.load(
        root=params_ctrl.path.saves_dir,
        model_place=models.controller,
    )
    p_pctrl: PControl
    p_pctrl.type(dtype)
    p_pctrl.to(device)
    p_pctrl.eval()

    model, _ = tool.util.load_direct(
        weight_path=saved_params_ctrl.path.used_nvae_weight, model_place=models.core
    )
    model: NewtonianVAEBase
    model.type(dtype)
    model.to(device)
    model.eval()

    path_result = params_ctrl.path.results_dir
    env = ControlSuiteEnvWrap(**params_ctrl.raw["ControlSuiteEnvWrap"])

    if steps is not None:
        max_time_length = steps
    else:
        max_time_length = params_ctrl.train.max_time_length

    del params_ctrl

    record, x_goal_n = calculate(
        saved_params_ctrl=saved_params_ctrl,
        p_pctrl=p_pctrl,
        model=model,
        env=env,
        episodes=episodes,
        time_length=max_time_length,
    )

    del saved_params_ctrl

    all_steps = max_time_length * episodes

    p = draw(
        record=record,
        x_goal_n=x_goal_n,
        max_time_length=max_time_length,
        all_steps=all_steps,
        fix_xmap_size=fix_xmap_size,
        env_domain=env.domain,
        env_task=env.task,
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
    saved_params_ctrl: paramsmanager.Params,
    p_pctrl: PControl,
    model: NewtonianVAEBase,
    env: ControlSuiteEnvWrap,
    episodes: int,
    time_length: int,
):
    torch.set_grad_enabled(False)

    record = Pack()

    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device

    s_u = saved_params_ctrl.preprocess.scale_u
    if s_u is not None:
        scaler_u = tp.Scaler(*s_u)
    else:
        scaler_u = tp.Scaler()

    s_x = saved_params_ctrl.preprocess.scale_x
    if s_x is not None:
        scaler_x = tp.Scaler(*s_x)
    else:
        scaler_x = tp.Scaler()

    # print(scaler_u)
    # print(scaler_x)

    print("Calculating...")
    for episode in range(episodes):
        record.add_list()

        cell_wrap = CellWrap(cell=model.cell)

        action = torch.zeros(size=(1, model.cell.dim_x), device=device, dtype=dtype)
        observation = env.reset().to(dtype=dtype).to(device=device)

        for t in range(time_length):
            action_prev = action  # for recording

            x_t, I_t_dec = cell_wrap.step(action=action_prev, observation=observation.unsqueeze(0))

            x_t_ = scaler_x.pre(x_t)
            action_ = p_pctrl.step(x_t_)  # u_t ~ P(u_t | x_t)
            action = scaler_u.post(action_)

            observation, _, done, position = env.step(action)
            observation = observation.to(dtype=dtype).to(device=device)

            record.append(
                action=to_numpy(action_prev.squeeze(0)),
                observation=to_numpy(observation.squeeze(0)),
                reconstruction=to_numpy(I_t_dec.squeeze(0)),
                x=to_numpy(x_t.squeeze(0)),
                position=position,
                pi=to_numpy(p_pctrl.param_pi.squeeze(0)),
            )

            Prompt.print_one_line(
                f"Trial: {episode+1:2d}/{episodes} | Step: {t+1:4d}/{time_length} "
            )

    print("\nDone")
    record.to_whole_np(show_shape=True)

    # x_goal_n = scaler_u.post(p_pctrl.param_mu.squeeze(0) / p_pctrl.K_n)
    x_goal_n = scaler_u.post(p_pctrl.x_goal_n)

    print("=== PControl params ===")
    Color.print("x^goal_n:")
    print(x_goal_n)
    Color.print("K_n")
    print(p_pctrl.K_n.detach().cpu())

    return record, x_goal_n


def draw(
    record: Pack,
    x_goal_n: Tensor,
    max_time_length: int,
    all_steps: int,
    fix_xmap_size: float = None,
    env_domain: str = None,
    env_task: str = None,
):
    dim_x = record.action.shape[-1]

    dim_colors = mpu.cmap(dim_x, "prism")
    label = Label(env_domain)

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
                range(dim_x),
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
            ax.set_xticks(range(dim_x))
            if env_domain == "reacher2d":
                ax.set_xlabel("Torque")
                ax.set_xticklabels([r"$\mathbf{u}[1]$ (shoulder)", r"$\mathbf{u}[2]$ (wrist)"])
            elif env_domain == "point_mass" and env_task == "easy":
                ax.set_xticklabels([r"$\mathbf{u}[1]$ (x)", r"$\mathbf{u}[2]$ (y)"])

            # ============================================================
            ax = axes.observation
            ax.set_title(r"$\mathbf{I}_t$ (From environment)")
            ax.imshow(obs2img(record.observation[self.episode, self.t]))
            ax.set_xlabel(f"{record.observation.shape[-1]} px")
            ax.set_ylabel(f"{record.observation.shape[-2]} px")
            ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)

            # ============================================================
            ax = axes.obs_dec
            ax.set_title(r"$\mathbf{I}_t$ (Reconstructed)")
            ax.imshow(obs2img(record.reconstruction[self.episode, self.t]))
            ax.set_xlabel(f"{record.reconstruction.shape[-1]} px")
            ax.set_ylabel(f"{record.reconstruction.shape[-2]} px")
            ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)

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
                x_goal_n[:, 0].cpu(),
                x_goal_n[:, 1].cpu(),
                marker="o",
                s=20,
                color="purple",
                label=r"$\mathbf{x}_{goal}$",
            )
            # ax.legend(loc="lower left")
            ax.legend()

            min_ = min(x_goal_n[:, 0].cpu().min(), record.x[..., 0].min())
            max_ = max(x_goal_n[:, 0].cpu().max(), record.x[..., 0].max())
            margin_ = (max_ - min_) * 0.1
            ax.set_xlim(min_ - margin_, max_ + margin_)

            min_ = min(x_goal_n[:, 1].cpu().min(), record.x[..., 1].min())
            max_ = max(x_goal_n[:, 1].cpu().max(), record.x[..., 1].max())
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
