import argparse
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils
import torch.utils.data
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec

import mypython.plot_config  # noqa: F401
import mypython.plotutil as mpu
import mypython.vision as mv
import tool.util
from models.core import NewtonianVAECell, NewtonianVAECellDerivation
from models.pcontrol import PurePControl
from mypython.terminal import Prompt
from simulation.env import ControlSuiteEnvWrap, img2obs, obs2img
from tool import argset
from tool.params import Params, ParamsEval, ParamsSimEnv
from tool.util import cmap_plt

parser = argparse.ArgumentParser(allow_abbrev=False)
argset.episodes(parser)
argset.anim_mode(parser)
argset.cf_simenv(parser)
argset.cf_eval(parser)
argset.path_model(parser)
argset.goal_img(parser)
argset.fix_xmap_size(parser)
argset.alpha(parser)
args = parser.parse_args()


def main():
    torch.set_grad_enabled(False)

    d = tool.util.select_date(args.path_model)
    if d is None:
        return
    weight_p = tool.util.select_weight(d)
    if weight_p is None:
        return

    params = Params(Path(d, "params_bk.json5"))
    params_eval = ParamsEval(args.cf_eval)
    params.general.steps = 300

    torch_dtype: torch.dtype = getattr(torch, params_eval.dtype)
    np_dtype: np.dtype = getattr(np, params_eval.dtype)

    if params_eval.device == "cuda" and not torch.cuda.is_available():
        print(
            "You have chosen cuda. But your environment does not support cuda, "
            "so this program runs on cpu."
        )
    device = torch.device(params_eval.device if torch.cuda.is_available() else "cpu")

    if params.general.derivation:
        cell = NewtonianVAECellDerivation(
            **params.newtonian_vae.kwargs, **params.newtonian_vae_derivation.kwargs
        )
    else:
        cell = NewtonianVAECell(**params.newtonian_vae.kwargs)

    cell.load_state_dict(torch.load(weight_p))
    cell.train(params_eval.training)
    cell.type(torch_dtype)
    cell.to(device)

    Path(params.path.result).mkdir(parents=True, exist_ok=True)

    all_steps = params.general.steps * args.episodes

    params_env = ParamsSimEnv(args.cf_simenv)
    env = ControlSuiteEnvWrap(**params_env.kwargs)

    Igoal = mv.cv2cnn(np.load(args.goal_img)).unsqueeze_(0)

    # ======
    fig = plt.figure()
    gs = GridSpec(nrows=5, ncols=6)
    up = 2
    axes: Dict[str, Axes] = {}
    axes["action"] = fig.add_subplot(gs[:up, 0:2])
    axes["observation"] = fig.add_subplot(gs[:up, 2:4])
    axes["Igoal"] = fig.add_subplot(gs[:up, 4:6])
    axes["x_map"] = fig.add_subplot(gs[up:5, :])

    class AnimPack:
        def __init__(self) -> None:
            self.t = -1

        def _attach_nan(self, x, i: int):
            xp = np
            return xp.concatenate(
                [
                    x[: self.t, i],
                    xp.full(
                        (params.general.steps - self.t,),
                        xp.nan,
                        dtype=x.dtype,
                    ),
                ]
            )

        def init(self):
            self.ctrl = PurePControl(img2obs(Igoal), args.alpha, cell)
            # print(self.ctrl.x_goal)
            self.observation = env.reset()
            xp = np
            self.LOG_x = xp.full(
                (params.general.steps, params.newtonian_vae.dim_x), xp.nan, dtype=np_dtype
            )

        def anim_func(self, frame_cnt):
            Prompt.print_one_line(
                f"{frame_cnt+1:5d} / {all_steps} ({(frame_cnt+1)*100/all_steps:.1f} %)"
            )

            mod = frame_cnt % params.general.steps
            if mod == 0:
                self.t = -1
                self.episode_cnt = frame_cnt // params.general.steps + mod

            # ======================================================
            self.t += 1
            color_action = tool.util.cmap_plt(params.newtonian_vae.dim_x, "prism")

            if frame_cnt == -1:
                self.episode_cnt = np.random.randint(0, args.episodes)
                self.t = params.general.steps - 1
                self.init()

            if self.t == 0:
                self.init()

            for ax in axes.values():
                ax.clear()

            fig.suptitle(
                f"Init environment: {self.episode_cnt+1}, step: {self.t:3d}",
                fontname="monospace",
            )

            x_t = cell.q_encoder.cond(self.observation).rsample()
            action = self.ctrl.get_action_from_x(x_t)
            # action = env.sample_random_action()
            self.observation, _, done, position = env.step(action)
            self.LOG_x[self.t] = x_t

            # ===============================================================
            ax = axes["action"]
            ax.set_title(r"$\mathbf{u}_{t-1}$ (Generated)")
            ax.set_aspect(aspect=0.7)
            ax.set_ylim(-1.2, 1.2)
            ax.bar(
                range(params.newtonian_vae.dim_x),
                action.detach().squeeze(0).cpu(),
                color=color_action,
                width=0.5,
            )
            ax.tick_params(bottom=False, labelbottom=False)

            # ===============================================================
            ax = axes["observation"]
            ax.set_title(r"$\mathbf{I}_t$")
            ax.imshow(mv.cnn2plt(obs2img(self.observation.detach().squeeze(0).cpu())))
            ax.set_axis_off()

            # ===============================================================
            ax = axes["Igoal"]
            ax.set_title(r"$\mathbf{I}_{goal}$")
            ax.imshow(mv.cnn2plt(Igoal.detach().squeeze(0).cpu()))
            ax.set_axis_off()

            # ===============================================================
            ax = axes["x_map"]
            ax.set_title(r"mean of $\mathbf{x}_{1:t}, \; \alpha = $" f"${args.alpha}$")
            ax.set_xlim(-args.fix_xmap_size, args.fix_xmap_size)
            ax.set_ylim(-args.fix_xmap_size, args.fix_xmap_size)
            ax.set_aspect(aspect=1)
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
                self.ctrl.x_goal.squeeze(0)[0],
                self.ctrl.x_goal.squeeze(0)[1],
                marker="o",
                s=20,
                color="purple",
                label=r"$\mathbf{x}_{goal}$",
            )
            ax.legend(loc="lower left")

            fig.tight_layout()

    p = AnimPack()

    version = 1
    while True:
        save_fname = f"control_{weight_p.parent.parent.stem}_W{weight_p.stem}_V{version}.mp4"
        s = Path(params.path.result, save_fname)
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
