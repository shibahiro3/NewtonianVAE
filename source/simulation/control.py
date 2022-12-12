import argparse
import sys
from pathlib import Path
from pprint import pprint
from typing import Dict

import json5
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter

import mypython.plotutil as mpu
import mypython.vision as mv
import tool.util
from models.core import (
    NewtonianVAECell,
    NewtonianVAEDerivationCell,
    NewtonianVAEV2Cell,
    get_NewtonianVAECell,
)
from models.pcontrol import PurePControl
from mypython.pyutil import Seq, add_version
from mypython.terminal import Color, Prompt
from newtonianvae.load import load
from simulation.env import ControlSuiteEnvWrap, img2obs, obs2img
from tool import argset, checker
from view.label import Label

try:
    import tool._plot_config

    figsize = tool._plot_config.figsize_control
except:
    figsize = None


parser = argparse.ArgumentParser(allow_abbrev=False)
argset.episodes(parser)
# argset.anim_mode(parser)
argset.save_anim(parser)
argset.cf_simenv(parser)
argset.cf_eval(parser)
argset.path_model(parser)
argset.path_result(parser)
argset.goal_img(parser)
argset.fix_xmap_size(parser)
argset.alpha(parser)
argset.steps(parser)
argset.env_domain(parser)
_args = parser.parse_args()


class Args:
    episodes = _args.episodes
    save_anim = _args.save_anim
    cf_simenv = _args.cf_simenv
    cf_eval = _args.cf_eval
    path_model = _args.path_model
    path_result = _args.path_result
    goal_img = _args.goal_img
    fix_xmap_size = _args.fix_xmap_size
    alpha = _args.alpha
    steps = _args.steps
    env_domain = _args.env_domain


args = Args()


def main():
    # =====================================================
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=1)
    mpu.get_figsize(fig)

    class Ax:
        def __init__(self) -> None:
            gs = GridSpec(nrows=5, ncols=12, hspace=0)
            up = 3

            v1 = Seq(step=3, start_now=True)
            v2 = Seq(step=4, start_now=True)

            # print(v1.now , v1.next() )
            # print(v1.now , v1.next() )
            # print(v1.now , v1.next() )

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
    label = Label(args.env_domain)
    # =====================================================

    torch.set_grad_enabled(False)
    if args.save_anim:
        checker.large_episodes(args.episodes)

    model, d, weight_path, params, params_eval, dtype, device = load(args.path_model, args.cf_eval)
    if weight_path is None:
        return

    assert type(model.cell) == NewtonianVAEV2Cell

    if args.steps is not None:
        max_time_length = args.steps
    else:
        max_time_length = params.train.max_time_length

    all_steps = max_time_length * args.episodes

    env = ControlSuiteEnvWrap(**json5.load(open(args.cf_simenv))["ControlSuiteEnvWrap"])

    Igoal = mv.cv2cnn(np.load(args.goal_img)).unsqueeze_(0).to(device)

    class AnimPack:
        def __init__(self) -> None:
            self.t = -1

        def _attach_nan(self, x, i: int):
            xp = np
            return xp.concatenate(
                [
                    x[: self.t, i],
                    xp.full(
                        (max_time_length - self.t,),
                        xp.nan,
                        dtype=x.dtype,
                    ),
                ]
            )

        def init(self):
            self.ctrl = PurePControl(img2obs(Igoal), args.alpha, model.cell)
            self.observation = env.reset().to(device)
            xp = np
            self.LOG_x = xp.full(
                (max_time_length, model.cell.dim_x),
                xp.nan,
                dtype=torch.empty((), dtype=dtype).numpy().dtype,
            )
            self.LOG_pos_0 = []
            self.LOG_pos_1 = []

        def anim_func(self, frame_cnt):
            axes.clear()

            Prompt.print_one_line(
                f"{frame_cnt+1:5d} / {all_steps} ({(frame_cnt+1)*100/all_steps:.1f} %)"
            )

            mod = frame_cnt % max_time_length
            if mod == 0:
                self.t = -1
                self.episode_cnt = frame_cnt // max_time_length + mod

            # ======================================================
            self.t += 1
            color_action = mpu.cmap(model.cell.dim_x, "prism")

            if frame_cnt == -1:
                self.episode_cnt = np.random.randint(0, args.episodes)
                self.t = max_time_length - 1
                self.init()

            if self.t == 0:
                self.init()

            fig.suptitle(
                f"Init environment: {self.episode_cnt+1}, step: {self.t:3d}",
                fontname="monospace",
            )

            # if self.t == 0:
            #     I_t_dec = torch.full_like(self.observation, torch.nan)
            #     x_t = model.cell.q_encoder.cond(self.observation).rsample()
            #     action = self.ctrl.get_action_from_x(x_t)
            #     v_t = torch.zeros_like(action)
            # else:
            #     I_t_dec, x_t, v_t = model.cell.decode(
            #         self.observation, self.x_tn1, self.action_pre, self.v_tn1, 0.1
            #     )
            #     action = self.ctrl.get_action_from_x(x_t)

            ### core ###
            x_t = model.cell.q_encoder.cond(self.observation).rsample()
            action = self.ctrl.get_action_from_x(x_t)
            I_t_dec = model.cell.p_decoder.cond(x_t).decode()
            ############

            # self.x_tn1 = x_t
            # self.action_pre = action
            # self.v_tn1 = v_t

            # action = env.sample_random_action()
            self.observation, _, done, position = env.step(action)
            self.observation = self.observation.to(device)
            self.LOG_x[self.t] = x_t.cpu()

            # ===============================================================
            ax = axes.action
            ax.set_title(r"$\mathbf{u}_{t-1}$ (Generated)")
            ax.set_ylim(-1.2, 1.2)
            ax.bar(
                range(model.cell.dim_x),
                action.detach().squeeze(0).cpu(),
                color=color_action,
                width=0.5,
            )
            ax.tick_params(bottom=False, labelbottom=False)
            mpu.Axis_aspect_2d(ax, 1)

            # ===============================================================
            ax = axes.observation
            ax.set_title(r"$\mathbf{I}_t$ (Env.)")
            ax.imshow(obs2img(self.observation.detach().cpu()))
            ax.set_axis_off()

            # ===============================================================
            ax = axes.obs_dec
            ax.set_title(r"$\mathbf{I}_t$ (Recon.)")
            ax.imshow(obs2img(I_t_dec.detach().squeeze(0).cpu()))
            ax.set_axis_off()

            # ===============================================================
            ax = axes.Igoal
            ax.set_title(r"$\mathbf{I}_{goal}$")
            ax.imshow(mv.cnn2plt(Igoal.detach().squeeze(0).cpu()))
            ax.set_axis_off()

            # ===============================================================
            ax = axes.x_map
            ax.set_title(r"$\mathbf{x}_{1:t}, \; \alpha = $" f"${args.alpha}$")
            ax.plot(
                self._attach_nan(self.LOG_x, 0),
                self._attach_nan(self.LOG_x, 1),
                marker="o",
                ms=2,
            )
            ax.scatter(
                self.LOG_x[self.t, 0],
                self.LOG_x[self.t, 1],
                marker="o",
                s=20,
                color="red",
                label=r"$\mathbf{x}_{t}$",
            )
            ax.scatter(
                self.ctrl.x_goal.squeeze(0)[0].cpu(),
                self.ctrl.x_goal.squeeze(0)[1].cpu(),
                marker="o",
                s=20,
                color="purple",
                label=r"$\mathbf{x}_{goal}$",
            )
            ax.legend(loc="lower left")
            label.set_axes_L0L1(ax, args.fix_xmap_size)

            # ===============================================================
            self.LOG_pos_0.append(position[0])

            ax = axes.x
            # ax.set_title(r"$\mathbf{x}_{1:t}, \; \alpha = $" f"${args.alpha}$")
            ax.plot(
                self.LOG_pos_0,
                self.LOG_x[: self.t + 1, 0],
                # self._attach_nan(self.LOG_x, 1),
                marker="o",
                ms=2,
            )
            ax.scatter(
                self.LOG_pos_0[-1],
                self.LOG_x[self.t, 0],
                marker="o",
                s=20,
                color="red",
                label=r"$\mathbf{x}_{t}$",
            )
            # ax.scatter(
            #     self.ctrl.x_goal.squeeze(0)[0].cpu(),
            #     self.ctrl.x_goal.squeeze(0)[1].cpu(),
            #     marker="o",
            #     s=20,
            #     color="purple",
            #     label=r"$\mathbf{x}_{goal}$",
            # )
            # ax.legend(loc="uppper left")
            label.set_axes_P0L0(ax, args.fix_xmap_size)

            # ===============================================================
            self.LOG_pos_1.append(position[1])

            ax = axes.y
            # ax.set_title(r"$\mathbf{x}_{1:t}, \; \alpha = $" f"${args.alpha}$")
            ax.plot(
                self.LOG_pos_1,
                self.LOG_x[: self.t + 1, 1],
                # self._attach_nan(self.LOG_x, 1),
                marker="o",
                ms=2,
            )
            ax.scatter(
                self.LOG_pos_1[-1],
                self.LOG_x[self.t, 1],
                marker="o",
                s=20,
                color="red",
                label=r"$\mathbf{x}_{t}$",
            )
            label.set_axes_P1L1(ax, args.fix_xmap_size)

    p = AnimPack()

    save_path = Path(args.path_result, f"{d.stem}_W{weight_path.stem}_control.mp4")
    save_path = add_version(save_path)
    mpu.anim_mode(
        "save" if args.save_anim else "anim",
        fig,
        p.anim_func,
        all_steps,
        interval=40,
        freeze_cnt=-1,
        save_path=save_path,
    )

    print()


if __name__ == "__main__":
    main()
