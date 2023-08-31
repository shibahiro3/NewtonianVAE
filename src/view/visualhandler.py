import pickle
from pprint import pprint
from typing import Any, Dict, List

import numpy as np
import requests
import visdom
from torch.utils.tensorboard import SummaryWriter

from view.visualhandlerbase import VisualHandlerBase


class TensorBoardVisualHandler(VisualHandlerBase):
    def __init__(self, *args, **kwargs):
        self.writer = SummaryWriter(*args, **kwargs)

    def plot(self, d: Dict[str, Any]):
        if d["mode"] == "all":
            losses = d["losses"]

            for loss_name in losses["train"].keys():
                self.writer.add_scalars(
                    f"{self.title}/Epoch Loss/{loss_name}",
                    {
                        "train": losses["train"][loss_name],
                        "valid": losses["valid"][loss_name],
                    },
                    d["epoch"],
                )

        self.writer.flush()


class VisdomVisualHandler(VisualHandlerBase):
    def __init__(self, *args, **kwargs):
        self.vis = visdom.Visdom(*args, **kwargs)

    def plot(self, d: Dict[str, Any]):
        if d["mode"] == "all":
            losses = d["losses"]

            for loss_name in losses["train"].keys():
                self.vis.line(
                    np.array([losses["train"][loss_name]]),
                    X=np.array([d["epoch"]]),
                    update="append",
                    win=loss_name,
                    name="train",
                    opts=dict(title=loss_name),
                )
                self.vis.line(
                    np.array([losses["valid"][loss_name]]),
                    X=np.array([d["epoch"]]),
                    update="append",
                    win=loss_name,
                    name="valid",
                    opts=dict(title=loss_name),
                )


class RequestsVisualHandler(VisualHandlerBase):
    def __init__(self, port):
        self._port = port

    def plot(self, d: Dict[str, Any]):
        if d["mode"] == "all":
            losses = d["losses"]
            try:
                requests.post(
                    url=f"http://localhost:{self._port}",
                    headers={"Content-Type": "application/octet-stream"},
                    data=pickle.dumps(losses),
                )
            except:
                pass
