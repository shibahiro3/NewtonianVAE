import sys

sys.path.append("../")

import argparse
from pathlib import Path
from typing import Dict

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import mypython.plot_config  # noqa: F401
import mypython.plotutil as mpu
import mypython.vision as mv
import numpy as np
import torch
import torch.utils
import torch.utils.data
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
from mypython.terminal import Prompt

import tool.argset as argset
import tool.util
from models.core import (
    CollectTimeSeriesData,
    NewtonianVAECell,
    NewtonianVAECellDerivation,
)
from simulation.env import obs2img
from tool.dataloader import GetBatchData
from tool.params import Params, ParamsEval

parser = argparse.ArgumentParser(allow_abbrev=False)
argset.episodes(parser)
argset.anim_mode(parser)
argset.cf_eval(parser)
argset.path_model(parser)
args = parser.parse_args()


def reconstruction():
    torch.set_grad_enabled(False)

    d = tool.util.select_date(args.path_model)
    if d is None:
        return
    weight_p = tool.util.select_weight(d)
    if weight_p is None:
        return

    params = Params(Path(d, "params_bk.json5"))
    params_eval = ParamsEval(args.cf_eval)
    print(params)
    print(params_eval)

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
    BatchData = GetBatchData(
        params.path.data,
        params_eval.data_start,
        params_eval.data_stop,
        args.episodes,
        dtype=torch_dtype,
    )
    action, observation = next(BatchData)

    # =======
    fig = plt.figure()
    gs = GridSpec(nrows=5, ncols=6)
    axes: Dict[str, Axes] = {}
    axes["action"] = fig.add_subplot(gs[0, 0:2])
    axes["observation"] = fig.add_subplot(gs[0, 2:4])
    axes["reconstructed"] = fig.add_subplot(gs[0, 4:6])
    axes["x_mean"] = fig.add_subplot(gs[1, :4])
    axes["x_dims"] = fig.add_subplot(gs[1, 4])
    axes["x_map"] = fig.add_subplot(gs[1, 5])
    axes["v"] = fig.add_subplot(gs[2, :4])
    axes["v_dims"] = fig.add_subplot(gs[2, 4])
    axes["xhat_mean"] = fig.add_subplot(gs[3, :4])
    axes["xhat_mean_dims"] = fig.add_subplot(gs[3, 4])
    axes["xhat_std"] = fig.add_subplot(gs[4, :4])
    axes["xhat_std_dims"] = fig.add_subplot(gs[4, 4])

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
            self.collector = CollectTimeSeriesData(
                dim_x=params.newtonian_vae.dim_x,
                dim_xhat=params.newtonian_vae_derivation.dim_xhat,
                T=params.general.steps,
                dtype=np_dtype,
            )
            # Igoal = mv.cv2cnn(np.load("observation_imgs/obs_2.npy")).unsqueeze_(0)
            # # print(Igoal.shape)
            # self.ctrl = PurePControl(Igoal, 1, cell)

            self.action, self.observation = (
                action[:, [self.episode_cnt]],
                observation[:, [self.episode_cnt]],
            )
            self.collector.run(
                cell,
                self.action,
                self.observation,
                params_eval.device,
                is_save=True,
            )

            # ----------------
            self.min_x_mean, self.max_x_mean = _min_max(self.collector.LOG_x_mean)

            self.min_x_mean_dims, self.max_x_mean_dims = _min_max(self.collector.LOG_x_mean, axis=0)
            # l = 12
            # self.min_x_mean_dims, self.max_x_mean_dims = np.array([[-l, -l], [l, l]])

            self.min_v, self.max_v = _min_max(self.collector.LOG_v)

            if type(cell) == NewtonianVAECellDerivation:
                self.min_xhat_mean, self.max_xhat_mean = _min_max(self.collector.LOG_xhat_mean)
                self.min_xhat_std, self.max_xhat_std = _min_max(self.collector.LOG_xhat_std)

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
                f"validational episode: {self.episode_cnt+1}, t = {self.t:3d}",
                fontname="monospace",
            )

            color_map = cm.get_cmap("rainbow")

            # ===============================================================
            ax = axes["action"]
            ax.set_title(r"$\mathbf{u}_{t-1}$ (Original)")
            # ax.set_xlabel("$u_x$")
            # ax.set_ylabel("$u_y$")
            ax.set_aspect(aspect=0.7)
            # ax.set_xlim(-1, 1)
            ax.set_ylim(-1.2, 1.2)
            # ax.arrow(0, 0, action[t, 0], action[t, 1], head_width=0.05)
            ax.bar(
                range(params.newtonian_vae.dim_x),
                CollectTimeSeriesData.as_save(self.action[self.t]),
                color=color_action,
                width=0.5,
            )
            ax.tick_params(bottom=False, labelbottom=False)

            # ===============================================================
            ax = axes["observation"]
            ax.set_title(r"$\mathbf{I}_t$ (Original)")
            ax.imshow(mv.cnn2plt(obs2img(CollectTimeSeriesData.as_save(self.observation[self.t]))))
            ax.set_axis_off()

            # ===============================================================
            ax = axes["reconstructed"]
            ax.set_title(r"$\mathbf{I}_t$ (Reconstructed)")
            ax.imshow(mv.cnn2plt(obs2img(self.collector.LOG_I_dec[self.t])))
            ax.set_axis_off()

            # ===============================================================
            ax = axes["x_mean"]
            N = params.newtonian_vae.dim_x
            ax.set_title(r"mean of $\mathbf{x}_{1:t}$")
            ax.set_xlim(0, params.general.steps)
            ax.set_ylim(self.min_x_mean, self.max_x_mean)
            ax.yaxis.set_major_formatter(FormatStrFormatter("%2.2f"))
            for i in range(N):
                ax.plot(
                    range(params.general.steps),
                    self._attach_nan(self.collector.LOG_x_mean, i),
                    color=color_map(1 - i / N),
                    lw=1,
                )

            # ===============================================================
            ax = axes["x_dims"]
            N = params.newtonian_vae.dim_x
            ax.set_title(r"mean of $\hat{\mathbf{x}}_{t}$  " f"(dim: {N})")
            ax.set_ylim(self.min_x_mean, self.max_x_mean)
            # ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%2.2f"))
            ax.bar(
                range(N),
                self.collector.LOG_x_mean[self.t],
                color=[color_map(1 - i / N) for i in range(N)],
            )
            ax.tick_params(bottom=False, labelbottom=False)

            # ===============================================================
            ax = axes["x_map"]
            ax.set_title(r"mean of $\mathbf{x}_{1:t}$")
            ax.set_xlim(self.min_x_mean_dims[0], self.max_x_mean_dims[0])
            ax.set_ylim(self.min_x_mean_dims[1], self.max_x_mean_dims[1])
            ax.set_aspect(aspect=1)
            # ax.set_xticks([self.min_x_mean, self.max_x_mean])
            # ax.set_yticks([self.min_x_mean, self.max_x_mean])
            # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            # ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.plot(
                self._attach_nan(self.collector.LOG_x_mean, 0),
                self._attach_nan(self.collector.LOG_x_mean, 1),
                marker="o",
                ms=2,
            )
            ax.plot(
                self.collector.LOG_x_mean[self.t, 0],
                self.collector.LOG_x_mean[self.t, 1],
                marker="o",
                ms=5,
                color="red",
            )

            # ===============================================================
            ax = axes["v"]
            N = params.newtonian_vae.dim_x
            ax.set_title(r"$\mathbf{v}_{1:t}$")
            ax.set_xlim(0, params.general.steps)
            ax.set_ylim(self.min_v, self.max_v)
            ax.yaxis.set_major_formatter(FormatStrFormatter("%2.2f"))
            for i in range(N):
                ax.plot(
                    range(params.general.steps),
                    self._attach_nan(self.collector.LOG_v, i),
                    color=color_map(1 - i / N),
                    lw=1,
                )

            # ===============================================================
            ax = axes["v_dims"]
            N = params.newtonian_vae.dim_x
            ax.set_title(r"$\mathbf{v}_t$  " f"(dim: {N})")
            ax.yaxis.set_major_formatter(FormatStrFormatter("%2.2f"))
            # ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_ylim(self.min_v, self.max_v)
            ax.bar(
                range(N),
                self.collector.LOG_v[self.t],
                color=[color_map(1 - i / N) for i in range(N)],
            )
            ax.tick_params(bottom=False, labelbottom=False)

            if type(cell) == NewtonianVAECellDerivation:
                # ===============================================================
                ax = axes["xhat_mean"]
                N = params.newtonian_vae_derivation.dim_xhat
                ax.set_title(r"mean of $\hat{\mathbf{x}}_{1:t}$")
                ax.set_xlim(0, params.general.steps)
                ax.set_ylim(self.min_xhat_mean, self.max_xhat_mean)
                ax.yaxis.set_major_formatter(FormatStrFormatter("%2.2f"))
                for i in range(N):
                    ax.plot(
                        range(params.general.steps),
                        self._attach_nan(self.collector.LOG_xhat_mean, i),
                        color=color_map(1 - i / N),
                        lw=1,
                    )

                # ===============================================================
                ax = axes["xhat_mean_dims"]
                N = params.newtonian_vae_derivation.dim_xhat
                ax.set_title(r"mean of $\hat{\mathbf{x}}_t$  " f"(dim: {N})")
                ax.yaxis.set_major_formatter(FormatStrFormatter("%2.2f"))
                ax.set_ylim(self.min_xhat_mean, self.max_xhat_mean)
                ax.bar(
                    range(N),
                    self.collector.LOG_xhat_mean[self.t],
                    color=[color_map(1 - i / N) for i in range(N)],
                )
                ax.tick_params(bottom=False, labelbottom=False)

                # ===============================================================
                ax = axes["xhat_std"]
                N = params.newtonian_vae_derivation.dim_xhat
                ax.set_title(r"std of $\hat{\mathbf{x}}_{1:t}$")
                ax.set_xlim(0, params.general.steps)
                ax.set_ylim(self.min_xhat_std, self.max_xhat_std)
                ax.yaxis.set_major_formatter(FormatStrFormatter("%2.2f"))
                for i in range(N):
                    ax.plot(
                        range(params.general.steps),
                        self._attach_nan(self.collector.LOG_xhat_std, i),
                        color=color_map(1 - i / N),
                        lw=1,
                    )

                # ===============================================================
                ax = axes["xhat_std_dims"]
                N = params.newtonian_vae_derivation.dim_xhat
                ax.set_title(r"std of $\hat{\mathbf{x}}_t$  " f"(dim: {N})")
                ax.set_ylim(self.min_xhat_std, self.max_xhat_std)
                ax.yaxis.set_major_formatter(FormatStrFormatter("%2.2f"))
                ax.bar(
                    range(N),
                    self.collector.LOG_xhat_std[self.t],
                    color=[color_map(1 - i / N) for i in range(N)],
                )
                ax.tick_params(bottom=False, labelbottom=False)

            fig.tight_layout()

    p = AnimPack()

    version = 1
    while True:
        save_fname = f"reconstructed_{weight_p.parent.parent.stem}_W{weight_p.stem}_V{version}.mp4"
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


def _min_max(x, axis=None, pad_ratio=0.2):
    min = np.nanmin(x, axis=axis)
    max = np.nanmax(x, axis=axis)
    padding = (max - min) * pad_ratio
    min -= padding
    max += padding
    return min, max


if __name__ == "__main__":
    reconstruction()
