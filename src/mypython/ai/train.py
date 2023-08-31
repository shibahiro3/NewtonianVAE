import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import ticker
from matplotlib.gridspec import GridSpec
from torch import Tensor, nn, optim

from .. import plotutil as mpu
from .. import rdict
from ..ai.util import BatchIndices
from ..numeric import MovingAverageTime
from ..pyutil import s2dhms_str
from ..terminal import Color, Prompt
from ..valuewriter import ValueWriter

# BatchDataType = Dict[str, Any]
BatchDataType = Any


class BaseModel(nn.Module):
    ReturnType = Union[Tensor, Tuple[Tensor, Dict[str, float]]]

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, *args, **kwargs) -> ReturnType:
        return super().__call__(*args, **kwargs)

    def forward(batchdata: BatchDataType) -> ReturnType:
        """
        Returns:
            L (Tensor)     -> L.backward()
            losses (dict)  -> Recording   (Optional)
        """
        raise NotImplementedError()


def train(
    *,
    model: BaseModel,
    optimizer: optim.Optimizer,
    trainloader: BatchIndices,
    epochs: int,
    #
    validloader: BatchIndices = None,
    unpack=True,  # model(input) or model(*input)
    save_per_epoch: Optional[int] = None,
    managed_dir: Union[str, Path, None] = None,
    grad_clip_norm: Optional[float] = None,
    pre_epoch_fn: Optional[Callable[[int], None]] = None,  # no_grad
    pre_batch_fn: Optional[Callable[[float, Any], Any]] = None,
    post_batch_fn: Optional[Callable[[float, dict], None]] = None,  # no_grad
    post_epoch_fn: Optional[Callable[[int, dict], None]] = None,  # no_grad
    print_precision: int = 4,
    gradscaler_args: Optional[dict] = None,
    use_autocast: bool = False,
) -> None:
    r"""

    all index start from 1

    pre_epoch_fn(epoch: int) -> None
    pre_batch_fn(epoch: float, *args) -> batchdata
    post_batch_fn(epoch: float, status: dict) -> None
        status: {
            "epoch": <int>,
            "losses": {"<loss name>": <float>, ...},
            "mode": "now",
            "phase": <"train" or "valid">
        }
    post_epoch_fn(epoch: int, status: dict) -> None
        status: {
            "epoch": <int>,
            "losses": { // Average per epoch
                "train": {"<loss name>": <float>, ...},
                "valid": {"<loss name>": <float>, ...}, // same names if validloader is not None
            },
            "mode": "all"
        }

    {managed_dir}
    ├── weight
    │   ├── {epoch}.pth
    │   ...


    Note:
        Hold-out Validation
    """

    if managed_dir is None:
        managed_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print("Auto created managed_dir:", managed_dir)
    else:
        _wp = Path(managed_dir, "weight")
        if len(list(_wp.glob("*"))) > 0:
            raise Exception(f"managed_dir weight has aleady weight files: {_wp}")

    managed_dir = Path(managed_dir)
    managed_dir.mkdir(parents=True, exist_ok=True)
    weight_dir = Path(managed_dir, "weight")
    weight_dir.mkdir(parents=True, exist_ok=True)

    record_losses: Dict[str, list] = {}
    batch_train_writer = ValueWriter(Path(managed_dir, "batch train"))
    epoch_train_writer = ValueWriter(Path(managed_dir, "epoch train"))
    epoch_valid_writer = ValueWriter(Path(managed_dir, "epoch valid"))

    if next(model.parameters()).device == torch.device("cpu"):
        Color.print("Warning: model is on cpu", c=Color.code.coral)

    if validloader is None:
        phases = ["train"]
    else:
        phases = ["train", "valid"]

    if gradscaler_args is not None:
        scaler = torch.cuda.amp.GradScaler(enabled=True, **gradscaler_args)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=False)

    try:
        epoch_f = 0.0
        time_start_learning = time.perf_counter()
        time_first_training = MovingAverageTime(5)
        time_epoch_training = MovingAverageTime(1)
        for epoch in range(1, epochs + 1):
            time_start_epoch = time.perf_counter()

            if pre_epoch_fn is not None:
                with torch.no_grad():
                    pre_epoch_fn(epoch)

            epoch_losses_all = {"train": {}, "valid": {}}
            for phase in phases:
                epoch_losses = {}

                if phase == "train":
                    torch.set_grad_enabled(True)
                    # torch.inference_mode(False)
                    # torch.autograd.detect_anomaly()
                    dataloader = trainloader
                    model.train()
                else:
                    torch.set_grad_enabled(False)
                    # torch.inference_mode(True)
                    dataloader = validloader
                    model.eval()

                ##### core #####
                for batch_i, batchdata in enumerate(dataloader, 1):
                    if pre_batch_fn is not None:
                        if unpack:
                            batchdata = pre_batch_fn(epoch_f, *batchdata)
                        else:
                            batchdata = pre_batch_fn(epoch_f, batchdata)

                    # https://h-huang.github.io/tutorials/recipes/recipes/amp_recipe.html
                    with torch.autocast("cuda", enabled=use_autocast):
                        if unpack:
                            output = model(*batchdata)
                        else:
                            output = model(batchdata)

                    if type(output) == Tensor:
                        L = output
                        losses = {}
                    elif len(output) == 2:
                        L, losses = output
                    else:
                        assert False

                    if phase == "train":
                        optimizer.zero_grad(set_to_none=True)
                        scaler.scale(L).backward()
                        # print_module_params(model, True)
                        if grad_clip_norm is not None:
                            # https://h-huang.github.io/tutorials/recipes/recipes/amp_recipe.html#inspecting-modifying-gradients-e-g-clipping
                            scaler.unscale_(optimizer)
                            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                        # optimizer.step()
                        scaler.step(optimizer)
                        scaler.update()
                    ##### end of core #####

                    # === show progress ===

                    L = L.cpu().item()

                    losses: Dict[str, float]
                    for k in losses.keys():
                        if type(losses[k]) == Tensor:
                            losses[k] = losses[k].cpu().item()

                    now_losses = {"Loss": L, **losses}
                    rdict.append_a_to_b(now_losses, record_losses)
                    rdict.add_a_to_b(now_losses, epoch_losses)

                    if phase == "train":
                        for k, v in now_losses.items():
                            batch_train_writer.write(k, v)

                        epoch_f = epoch - 1 + batch_i / len(dataloader)

                    # Color.print(f"epoch: {epoch}, {epoch_f:.3f}")

                    bt_msg = (
                        f"{phase} | Epoch: {epoch} | "
                        + " | ".join(
                            [f"{k}: {v:.{print_precision}f}" for k, v in now_losses.items()]
                        )
                        + f" | Elapsed: {s2dhms_str(time.perf_counter() - time_start_learning)} "
                    )

                    if epoch == 1:
                        if phase == "train":
                            remaining = time_first_training.update() * epochs * len(trainloader)

                        bt_msg += (
                            f"| Remaining: {s2dhms_str(remaining)}+ | "
                            + f"ETA: "
                            + (datetime.now() + timedelta(seconds=remaining)).strftime(
                                "%m/%d %H:%M+ "
                            )
                        )

                    else:
                        remaining = time_epoch_training.get() * (epochs - epoch + 1)
                        bt_msg += (
                            f"| Remaining: {s2dhms_str(remaining)}+ | "
                            + f"ETA: "
                            + (datetime.now() + timedelta(seconds=remaining)).strftime(
                                "%m/%d %H:%M+ "
                            )
                        )

                    Prompt.print_one_line(bt_msg)
                    if post_batch_fn is not None:
                        status = {
                            "mode": "now",
                            "epoch": epoch,
                            "losses": now_losses,
                            "phase": phase,
                        }
                        with torch.no_grad():
                            post_batch_fn(epoch_f, status)

                ### end of one epoch ###

                for k, v in epoch_losses.items():
                    epoch_losses[k] = v / len(dataloader)

                ep_msg = f"{phase} | Epoch: {epoch} | " + " | ".join(
                    [f"{k}: {v :.{print_precision}f}" for k, v in epoch_losses.items()]
                )
                ep_msg = Prompt.del_line + ep_msg

                if phase == "train":
                    epoch_f = epoch

                    if (save_per_epoch is not None) and (epoch % save_per_epoch == 0):
                        p_ = Path(weight_dir, f"{epoch}.pth")
                        p_.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(model.state_dict(), p_)
                        ep_msg += Color.green + " saved" + Color.reset
                else:
                    ep_msg = (
                        Color.code.coral
                        + ep_msg
                        + Color.reset
                        + f" | Duration: {s2dhms_str(time.perf_counter() - time_start_epoch)}"
                        + f" | Elapsed: {s2dhms_str(time.perf_counter() - time_start_learning)}"
                    )

                if validloader is None:
                    ep_msg = (
                        ep_msg
                        + f" | Duration: {s2dhms_str(time.perf_counter() - time_start_epoch)}"
                        + f" | Elapsed: {s2dhms_str(time.perf_counter() - time_start_learning)}"
                    )

                epoch_losses_all[phase] = epoch_losses

                print(ep_msg)  # average

            for k, v in epoch_losses_all["train"].items():
                epoch_train_writer.write(k, v)

            for k, v in epoch_losses_all["valid"].items():
                epoch_valid_writer.write(k, v)

            if post_epoch_fn is not None:
                status = {
                    "mode": "all",
                    "epoch": epoch,
                    "losses": epoch_losses_all,
                }
                with torch.no_grad():
                    post_epoch_fn(epoch, status)

            time_epoch_training.update()

    except KeyboardInterrupt:
        print(Prompt.cursor_down(2))
        print("KeyboardInterrupt")
    except:
        print(Prompt.cursor_down(2))
        print("=== traceback ===")
        print(traceback.format_exc())


def show_loss(
    *,
    manage_dir: Union[str, Path],
    results_dir: Union[str, Path],
    start_iter: int = 1,
    format: List[str] = ["pdf", "png"],
    mode: str = "epoch",
    no_window: bool = False,
):
    assert start_iter > 0
    assert mode in ("batch", "epoch")

    # plt.rcParams.update(
    #     {
    #         "figure.figsize": (11.39, 3.9),
    #         "figure.subplot.left": 0.05,
    #         "figure.subplot.right": 0.98,
    #         "figure.subplot.bottom": 0.15,
    #         "figure.subplot.top": 0.85,
    #         "figure.subplot.wspace": 0.4,
    #     }
    # )

    manage_dir = Path(manage_dir)
    day_time = manage_dir.stem
    datetime.strptime(day_time, "%Y-%m-%d_%H-%M-%S")  # for check (ValueError)

    if mode == "batch":
        losses = ValueWriter.load(Path(manage_dir, "batch train"))
    elif mode == "epoch":
        losses = ValueWriter.load(Path(manage_dir, "epoch train"))
        losses_valid = ValueWriter.load(Path(manage_dir, "epoch valid"))
    else:
        assert False

    keys = list(losses.keys())
    fig, axes = plt.subplots(1, len(keys))
    mpu.get_figsize(fig)
    fig.suptitle("Loss")

    alpha = 0.5
    start_idx = start_iter - 1

    def plot_axes(losses_, ax: plt.Axes, k, color, label=None):
        steps = len(losses_[k])
        assert start_idx < steps
        span = (steps - start_idx) // 20
        if span < 1:
            span = 1

        data = losses_[k][start_idx:]
        smooth = pd.DataFrame(data).ewm(span=span).mean()
        R = range(start_iter, steps + 1)
        ax.set_title(k)
        ax.plot(R, data, color=color, alpha=alpha)
        ax.plot(R, smooth, color=color, lw=2, label=label)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        # ax.set_xticks(np.linspace(start_iter, steps, 5, dtype=int))  # OK: len(data) < 5
        ax.grid(ls=":")

    if mode == "batch":
        for i, k in enumerate(keys):
            plot_axes(losses, axes[i], k, color="dodgerblue")
        fig.text(0.5, 0.03, "Iterations", ha="center", va="center", fontsize=14)

    elif mode == "epoch":
        for i, k in enumerate(keys):
            plot_axes(losses, axes[i], k, color="dodgerblue", label="train")
            plot_axes(losses_valid, axes[i], k, color="orange", label="valid")
        axes[-1].legend()
        fig.text(0.5, 0.03, "Epochs", ha="center", va="center", fontsize=14)

    # mpu.legend_reduce(fig, loc="lower right")

    save_path = save_pathname(root=results_dir, day_time=day_time, descr="loss")
    if no_window:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        for fmt in format:
            p = save_path.with_suffix(f".{fmt}")
            plt.savefig(p)
            Color.print("saved to:", p)
    else:
        mpu.register_save_path(fig, save_path, format)
        plt.show()


def save_pathname(*, root, day_time: str, descr, epoch=None):
    """Put the execution date and time on the file name"""
    # .with_suffix(f".{fmt}")

    datetime.strptime(day_time, "%Y-%m-%d_%H-%M-%S")  # for check (ValueError)

    if epoch is None:
        # save_path = Path(root, day_time, f"{day_time}_{descr}")
        save_path = Path(root, f"{day_time}_{descr}")
    else:
        # save_path = Path(root, day_time, f"{day_time}_E{epoch}_{descr}")
        save_path = Path(root, f"{day_time}_E{epoch}_{descr}")

    return save_path
