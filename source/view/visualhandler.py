from typing import Any, Dict, List

import numpy as np
import visdom
from torch.utils.tensorboard import SummaryWriter

from view.visualhandlerbase import VisualHandlerBase


class TensorBoardVisualHandler(VisualHandlerBase):
    def __init__(self, *args, **kwargs):
        self.writer = SummaryWriter(*args, **kwargs)

        self.step = 0

    def plot(self, d: Dict[str, Any]):
        for k, v in d.items():
            if "Loss" not in k:
                continue

            self.writer.add_scalar(k, v, self.step)

        self.step += 1


class VisdomVisualHandler(VisualHandlerBase):
    def __init__(self, *args, **kwargs):
        self.vis = visdom.Visdom(*args, **kwargs)

        self.step = 0

    def plot(self, d: Dict[str, Any]):
        for k, v in d.items():
            if "Loss" not in k:
                continue

            self.vis.line(
                np.array([v]),
                X=np.array([self.step]),
                update="append",
                win=k,
                opts={"title": k},
            )

        self.step += 1
