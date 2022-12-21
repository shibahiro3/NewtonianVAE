import sys
from pathlib import Path
from pprint import pprint
from typing import Dict, Union

import classopt
import json5
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter

import models.core
import mypython.plotutil as mpu
import mypython.vision as mv
import tool.plot_config
import tool.util
from models.cell import CellWrap
from models.core import NewtonianVAEFamily
from models.pcontrol import PurePControl
from mypython.pyutil import Seq, add_version
from mypython.terminal import Color, Prompt
from simulation.env import ControlSuiteEnvWrap, img2obs, obs2img
from tool import argset, checker, paramsmanager
from view.label import Label

tool.plot_config.apply()
try:
    import tool._plot_config

    tool._plot_config.apply()
except:
    pass

config = {
    "figure.figsize": (12.76, 8.39),
    "figure.subplot.left": 0.05,
    "figure.subplot.right": 0.95,
    "figure.subplot.bottom": 0.1,
    "figure.subplot.top": 1,
    "figure.subplot.hspace": 0.3,
    "figure.subplot.wspace": 0.3,
}
plt.rcParams.update(config)


@classopt.classopt(default_long=True, default_short=False)
class Args:
    config: str = classopt.config(**argset.descr_config, required=True)
    episodes: int = classopt.config(**argset.descr_episodes, required=True)
    path_model: str = classopt.config(**argset.descr_path_model, required=False)
    path_result: str = classopt.config(**argset.descr_path_result, required=False)
    fix_xmap_size: float = classopt.config(metavar="S", help="xmap size")
    save_anim: bool = classopt.config()
    goal_img: str = classopt.config(metavar="PATH", help="Goal image path (*.npy)")
    alpha: float = classopt.config(metavar="α", help="P gain")
    steps: int = classopt.config(metavar="E", help="Time steps")
    env_domain: str = classopt.config(metavar="ENV")
    movie_format: str = classopt.config(default="mp4")


args = Args.from_args()  # pylint: disable=E1101


def main():
    # =====================================================
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
    label = Label(args.env_domain)
    # =====================================================

    torch.set_grad_enabled(False)
    if args.save_anim:
        checker.large_episodes(args.episodes)

    _params = paramsmanager.Params(args.config)
    params_eval = _params.eval
    path_model = tool.util.priority(args.path_model, _params.external.save_path)
    del _params
    path_result = tool.util.priority(args.path_result, params_eval.result_path)

    dtype, device = tool.util.dtype_device(
        dtype=params_eval.dtype,
        device=params_eval.device,
    )

    model, manage_dir, weight_path, params = tool.util.load(
        root=path_model, model_place=models.core
    )

    model: NewtonianVAEFamily
    model.type(dtype)
    model.to(device)
    model.train(params_eval.training)

    if args.steps is not None:
        max_time_length = args.steps
    else:
        max_time_length = params.train.max_time_length

    all_steps = max_time_length * args.episodes

    env = ControlSuiteEnvWrap(**json5.load(open(args.config))["ControlSuiteEnvWrap"])

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
            self.cell_wrap = CellWrap(cell=model.cell)

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
                self.action = torch.zeros(size=(1, model.cell.dim_x), device=device, dtype=dtype)
                self.observation = env.reset().to(device)

            fig.suptitle(
                f"Init environment: {self.episode_cnt+1}, step: {self.t:3d}",
                fontname="monospace",
            )

            ### core ###
            x_t, I_t_dec = self.cell_wrap.step(action=self.action, observation=self.observation)
            self.action = self.ctrl.step(x_t)
            ############

            self.observation, _, done, position = env.step(self.action)
            self.observation = self.observation.to(device)
            self.LOG_x[self.t] = x_t.cpu()

            # ===============================================================
            ax = axes.action
            ax.set_title(r"$\mathbf{u}_{t-1}$ (Generated)")
            ax.set_ylim(-1.2, 1.2)
            ax.bar(
                range(model.cell.dim_x),
                self.action.detach().squeeze(0).cpu(),
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

    save_path = Path(
        path_result, f"{manage_dir.stem}_W{weight_path.stem}_control.{args.movie_format}"
    )
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
