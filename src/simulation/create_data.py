import pickle
import shutil
import sys
import time
from collections import ChainMap
from numbers import Number
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from matplotlib.gridspec import GridSpec
from torch import Tensor

import mypython.plotutil as mpu
import mypython.vision as mv
import tool.util
import view.plot_config
from mypython import rdict
from mypython.terminal import Color, Prompt
from simulation.env import ControlSuiteEnvWrap, obs2img
from third import json5
from tool import checker
from view.show_data import ShowData


def main(
    config: str,
    episodes: int,
    # watch: str,
    # save_anim: bool,
    mode: str = "plt",
    save_dir: Optional[str] = None,
    format: str = "mp4",
):
    assert mode in ("show-plt", "show-render", "save-data", "save-anim")

    view.plot_config.apply()

    if mode.startswith("save"):
        if save_dir is None:
            raise ValueError(f'You should set "--save-dir" if you set mode to "{mode}"')

    if mode == "save-anim":
        checker.large_episodes(episodes)

    with open(config, "r") as f:
        param_env = json5.load(f)["ControlSuiteEnvWrap"]

    # params = paramsmanager.Params(config, exclusive_keys=["ControlSuiteEnvWrap"])

    env = ControlSuiteEnvWrap(**param_env)
    T = env.max_episode_length // env.action_repeat
    all_steps = T * episodes

    print("observation size:", env.observation_size)
    print("action size:", env.action_size)
    print("action range:", env.action_range)

    if save_dir is not None:
        save_dir: Path = Path(save_dir)
        if mode == "save-data":
            if len(list(save_dir.glob("**/*.pickle"))) > 0:
                print(
                    f'\nIt seems "{save_dir}" already has data. This directory will be erased and replaced with new data.'
                )
                if input("Do you want to continue? [y/n] ") != "y":
                    print("Abort.")
                    return
                else:
                    shutil.rmtree(save_dir)

            save_dir.mkdir(parents=True, exist_ok=True)
            with open(Path(save_dir, "params_env_bk.json5"), "w") as f:
                json5.dump(param_env, f, indent=2)

    observations = env.reset()
    plt_render = ShowData(
        win_title="Create Data",
        n_cam=len(observations["camera"]),
        env_domain=env.domain,
        env_task=env.task,
        env_position_wrap=env.position_wrap,
    )

    class AnimPack:
        def __init__(self) -> None:

            self.t = 0
            self.episode_cnt = 0

        def anim_func(self, frame_cnt):
            if mode == "save-anim":
                Prompt.print_one_line(
                    f"{frame_cnt+1:5d} / {all_steps} ({(frame_cnt+1)*100/all_steps:.1f} %) "
                )

            mod = frame_cnt % T
            if mod == 0:
                env.reset()
                plt_render.first_episode()

                self.episode_data = {}

                self.t = 0
                self.episode_cnt = frame_cnt // T + mod
            else:
                self.t += 1

            # ======================================================

            ### core ###
            action = env.sample_random_action()
            # action = env.zeros_action()
            observations, done = env.step(action)

            step_data = observations
            step_data["action"] = action
            step_data["delta"] = 0.1

            rdict.to_numpy(step_data, ignore_scalar=True)
            rdict.append_a_to_b(step_data, self.episode_data)
            # rdict.show(step_data, "step_data")
            # rdict.show(self.episode_data, "episode_data")

            if mode == "show-render":
                env.render()

            elif mode == "show-plt" or mode == "save-anim":
                # Relative Position
                # relative_position
                plt_render.frame(
                    t=self.t + 1,
                    episode_cnt=self.episode_cnt + 1,
                    action=action,
                    Ist=observations["camera"],
                    position=observations["position"],
                    position_title="Position",
                    set_lim_fn=lambda p: p.ax_action.ax.set_ylim(-1.2, 1.2),
                )

            if done:

                # rdict.to_numpy(self.episode_data)
                # rdict.show(self.episode_data, "episode_data (save)")

                if mode != "save-anim":
                    if mode == "save-data":
                        episodes_dir = Path(save_dir, "episodes")
                        episodes_dir.mkdir(parents=True, exist_ok=True)

                        rdict.to_numpy(self.episode_data)
                        # rdict.show(self.episode_data, "episode_data (save)")
                        with open(Path(episodes_dir, f"{self.episode_cnt}.pickle"), "wb") as f:
                            pickle.dump(self.episode_data, f)

                        info = Color.green + "saved" + Color.reset
                    else:
                        info = Color.coral + "not saved" + Color.reset

                    Prompt.print_one_line(f"[{info}] Episode: {self.episode_cnt+1}, T = {self.t+1}")

    p = AnimPack()

    if mode == "show-plt":
        mpu.anim_mode("anim", plt_render.fig, p.anim_func, T * episodes, interval=40)

    elif mode == "save-anim":
        mpu.anim_mode(
            "save",
            plt_render.fig,
            p.anim_func,
            T * episodes,
            interval=40,
            save_path=Path(save_dir, f"data.{format}"),
        )

    else:
        for frame_cnt in range(T * episodes):
            p.anim_func(frame_cnt)

    print()
