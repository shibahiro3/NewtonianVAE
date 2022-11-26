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
from mypython.plotutil import Axis_aspect_2d, cmap
from mypython.terminal import Color, Prompt
from newtonianvae.load import load
from simulation.env import ControlSuiteEnvWrap, img2obs, obs2img
from tool import argset, checker

try:
    import tool._plot_config

    figsize = tool._plot_config.figsize_control
except:
    figsize = None


parser = argparse.ArgumentParser(allow_abbrev=False)
argset.episodes(parser)
argset.anim_mode(parser)
argset.cf_simenv(parser)
argset.cf_eval(parser)
argset.path_model(parser)
argset.path_result(parser)
argset.goal_img(parser)
argset.fix_xmap_size(parser)
argset.alpha(parser)
argset.steps(parser)
_args = parser.parse_args()


class Args:
    episodes = _args.episodes
    anim_mode = _args.anim_mode
    cf_simenv = _args.cf_simenv
    cf_eval = _args.cf_eval
    path_model = _args.path_model
    path_result = _args.path_result
    goal_img = _args.goal_img
    fix_xmap_size = _args.fix_xmap_size
    alpha = _args.alpha
    steps = _args.steps


args = Args()


def main():
    if args.anim_mode == "save":
        checker.large_episodes(args.episodes)

    torch.set_grad_enabled(False)

    model, d, weight_p, params, params_eval, dtype, device = load(args.path_model, args.cf_eval)
    if weight_p is None:
        return

    assert type(model.cell) == NewtonianVAEV2Cell

    Path(args.path_result).mkdir(parents=True, exist_ok=True)

    if args.steps is not None:
        max_time_length = args.steps
    else:
        max_time_length = params.train.max_time_length

    all_steps = max_time_length * args.episodes

    env = ControlSuiteEnvWrap(**json5.load(open(args.cf_simenv))["ControlSuiteEnvWrap"])

    Igoal = mv.cv2cnn(np.load(args.goal_img)).unsqueeze_(0).to(device)

    # ======
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=1)
    mpu.get_figsize(fig)

    class Ax:
        def __init__(self) -> None:
            gs = GridSpec(nrows=5, ncols=8, hspace=0)
            up = 3

            self.action = fig.add_subplot(gs[:up, 0:2])
            self.Igoal = fig.add_subplot(gs[:up, 2:4])
            self.observation = fig.add_subplot(gs[:up, 4:6])
            self.obs_dec = fig.add_subplot(gs[:up, 6:8])
            self.x_map = fig.add_subplot(gs[up:5, :])

        def clear(self):
            for ax in self.__dict__.values():
                ax.clear()

    axes = Ax()

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
            color_action = cmap(model.cell.dim_x, "prism")

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
            Axis_aspect_2d(ax, 1)

            # ===============================================================
            ax = axes.observation
            ax.set_title(r"$\mathbf{I}_t$ (Env.)")
            ax.imshow(mv.cnn2plt(obs2img(self.observation.detach().cpu())))
            ax.set_axis_off()

            # ===============================================================
            ax = axes.obs_dec
            ax.set_title(r"$\mathbf{I}_t$ (Recon.)")
            ax.imshow(mv.cnn2plt(obs2img(I_t_dec.detach().squeeze(0).cpu())))
            ax.set_axis_off()

            # ===============================================================
            ax = axes.Igoal
            ax.set_title(r"$\mathbf{I}_{goal}$")
            ax.imshow(mv.cnn2plt(Igoal.detach().squeeze(0).cpu()))
            ax.set_axis_off()

            # ===============================================================
            ax = axes.x_map
            ax.set_title(r"$\mathbf{x}_{1:t}, \; \alpha = $" f"${args.alpha}$")
            ax.set_xlim(-args.fix_xmap_size, args.fix_xmap_size)
            ax.set_ylim(-args.fix_xmap_size, args.fix_xmap_size)
            # ax.set_xticks([self.min_x_mean, self.max_x_mean])
            # ax.set_yticks([self.min_x_mean, self.max_x_mean])
            # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            # ax.yaxis.set_major_locator(MaxNLocator(integer=True))
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
            Axis_aspect_2d(ax, 1)

    p = AnimPack()

    version = 1
    while True:
        save_fname = f"control_{weight_p.parent.parent.stem}_W{weight_p.stem}_V{version}.mp4"
        s = Path(args.path_result, save_fname)
        version += 1
        if not s.exists():
            break

    if args.anim_mode == "save":
        print(f"save to: {s}")

    mpu.anim_mode(
        args.anim_mode,
        fig,
        p.anim_func,
        all_steps,
        interval=40,
        freeze_cnt=-1,
        save_path=s,
    )

    print()


if __name__ == "__main__":
    main()
