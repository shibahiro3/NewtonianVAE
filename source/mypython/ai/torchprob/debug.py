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
            submodel: Distribution
            if submodel._cnt_given == 0:
                color = Color.blue
            elif submodel._cnt_given > 1:
                color = Color.coral
            else:
                color = ""

            print(
                "name, times .given() has run = "
                + color
                + f"{name}, {submodel._cnt_given}"
                + Color.reset
            )
            submodel.clear_dist_parameters()
