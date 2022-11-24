from pprint import pprint
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from mypython.terminal import Color

from .distributions.base import Distribution


def check_dist_model(model: nn.Module):
    print("===== check_dist_model =====")

    for name, submodel in model._modules.items():  # __dict__ には無い
        if issubclass(type(submodel), Distribution):
            if submodel._cnt == 0:
                color = Color.blue
            elif submodel._cnt > 1:
                color = Color.coral
            else:
                color = ""

            print(
                "name, times .cond() has run = " + color + f"{name}, {submodel._cnt}" + Color.reset
            )
            submodel._cnt = 0
