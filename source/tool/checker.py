import sys

import torch


def large_episodes(episodes) -> None:
    if episodes > 10:
        print(
            (
                f"You have specified a large number of episodes (= {episodes})"
                ", though you are trying to save movie."
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
