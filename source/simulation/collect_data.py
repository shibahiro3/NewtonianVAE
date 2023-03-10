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
from tool import checker, paramsmanager
from tool.util import Preferences
from view.show_data import ShowData

view.plot_config.apply()
try:
    import view._plot_config

    view._plot_config.apply()
except:
    pass


def main(
    config: str,
    episodes: int,
    watch: str,
    save_anim: bool,
    format: str,
):
    if save_anim and watch != "plt":
        Color.print(
            "Ignore --save-anim option: Use --watch=plt option to save videos", c=Color.coral
        )

    if save_anim and watch == "plt":
        checker.large_episodes(episodes)

    params = paramsmanager.Params(config)

    env = ControlSuiteEnvWrap(**params.raw["ControlSuiteEnvWrap"])
    T = env.max_episode_length // env.action_repeat
    all_steps = T * episodes

    print("observation size:", env.observation_size)
    print("action size:", env.action_size)
    print("action range:", env.action_range)

    data_path = Path(params.path.data_dir)
    if watch is None:
        if len(list(data_path.glob("episodes/*"))) > 0:
            print(
                f'\n"{data_path}" already has data. This directory will be erased and replaced with new data.'
            )
            if input("Do you want to continue? [y/n] ") != "y":
                print("Abort.")
                return
            shutil.rmtree(data_path)

        data_path.mkdir(parents=True, exist_ok=True)
        params.save_simenv(Path(data_path, "params_env_bk.json5"))
        Preferences.put(data_path, "id", int(time.time() * 1000))

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
            if save_anim:
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

            if watch == "render":
                env.render()

            elif watch == "plt":
                # Relative Position
                # relative_position
                plt_render.frame(
                    t=self.t + 1,
                    episode_cnt=self.episode_cnt + 1,
                    action=action,
                    Ist=observations["camera"],
                    position=observations["position"],
                    position_title="Position",
                    set_lim=lambda p: p.ax_action.set_ylim(-1.2, 1.2),
                )

            if done and not save_anim:

                # rdict.to_numpy(self.episode_data)
                # rdict.show(self.episode_data, "episode_data (save)")

                if watch is None:
                    episodes_dir = Path(data_path, "episodes")
                    episodes_dir.mkdir(parents=True, exist_ok=True)

                    rdict.to_numpy(self.episode_data)
                    # rdict.show(self.episode_data, "episode_data (save)")
                    with open(Path(episodes_dir, f"{self.episode_cnt}.pickle"), "wb") as f:
                        pickle.dump(self.episode_data, f)

                    done_ = Color.green + "saved" + Color.reset
                else:
                    done_ = Color.coral + "not saved" + Color.reset

                Prompt.print_one_line(f"[{done_}] Episode: {self.episode_cnt+1}, T = {self.t+1}")

    p = AnimPack()

    if watch == "plt":
        save_path = Path(data_path, f"data.{format}")
        mpu.anim_mode(
            "save" if save_anim else "anim",
            plt_render.fig,
            p.anim_func,
            T * episodes,
            interval=40,
            save_path=save_path,
        )

    else:
        for frame_cnt in range(T * episodes):
            p.anim_func(frame_cnt)

    print()


if __name__ == "__main__":
    main()
