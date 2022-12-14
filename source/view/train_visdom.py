import numpy as np
import visdom

import newtonianvae.train
from tool.visualhandlerbase import VisualHandlerBase


class VisualHandler(VisualHandlerBase):
    def __init__(self):
        self.vis = visdom.Visdom()

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


if __name__ == "__main__":
    newtonianvae.train.train(VisualHandler())
