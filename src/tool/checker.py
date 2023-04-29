import sys
from pathlib import Path

import torch

from mypython.terminal import Color
from third import json5


def large_episodes(episodes) -> None:
    if episodes > 10:
        print(
            (
                f"You have specified a large number of episodes (= {episodes})"
                ", though you are trying to save video."
            )
        )
        input_ = input("Do you want to continue? [y/n] ")
        if input_ != "y":
            sys.exit()


def cuda(device):
    if device == "cuda" and not torch.cuda.is_available():
        print(
            "You have chosen cuda. But your environment does not support cuda, "
            "so this program runs on cpu."
        )


def is_same_data(data, trained_time_dir):
    data_p = Path(data).resolve()
    init_info_p = Path(trained_time_dir, "init_info.json5")
    if init_info_p.exists():
        with open(init_info_p) as f:
            trained_data_p = Path(json5.load(f)["data_from"]).resolve()
        if data_p != trained_data_p:
            Color.print(
                "warning: The path of the specified data does not match the path of the data used for training",
                c=Color.coral,
            )
            Color.print(f"  specified data        : {data_p}", c=Color.coral)
            Color.print(f"  data used for training: {trained_data_p}", c=Color.coral)
