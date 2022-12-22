import numpy as np
import visdom
from torch.utils.tensorboard import SummaryWriter

from tool.visualhandlerbase import VisualHandlerBase


class VisualHandlerBase:
    def __init__(self):
        self.title = ""

    def plot(self, *args, **kwargs) -> None:
        pass

    def wait_init(self):
        pass

    def call_end_init(self):
        pass

    def call_end(self) -> None:
        pass

    @property
    def is_running(self) -> bool:
        return True


class TensorBoardVisualHandler(VisualHandlerBase):
    def __init__(self, *args, **kwargs):
        self.writer = SummaryWriter(*args, **kwargs)

        self.step = 0

    def plot(self, L, E_ll, E_kl, epoch):
        self.writer.add_scalar("Loss", L, self.step)
        self.writer.add_scalar("NLL", E_ll, self.step)
        self.writer.add_scalar("KL", E_kl, self.step)
        self.step += 1


class VisdomVisualHandler(VisualHandlerBase):
    def __init__(self, *args, **kwargs):
        self.vis = visdom.Visdom(*args, **kwargs)

        self.step = 0

    def plot(self, L, E_ll, E_kl, epoch):
        self.vis.line(
            np.array([L]),
            X=np.array([self.step]),
            update="append",
            win="Loss",
            opts={"title": "Loss"},
        )
        self.vis.line(
            np.array([E_ll]),
            X=np.array([self.step]),
            update="append",
            win="NLL",
            opts={"title": "NLL"},
        )
        self.vis.line(
            np.array([E_kl]),
            X=np.array([self.step]),
            update="append",
            win="KL",
            opts={"title": "KL"},
        )
        self.step += 1
