from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch import Tensor

from mypython.ai.util import BatchIdx


class GetBatchData(BatchIdx):
    def __init__(
        self,
        path: Union[str, Path],
        startN: int,
        stopN: int,
        BS: int,
        dtype: torch.dtype,
        device=torch.device("cpu"),
    ):
        """
        path: directory path of data
            path
            ├── 0
            │   ├── action.npy
            │   └── observation.npy
            ├── 1
            │   ├── action.npy
            │   └── observation.npy
            ...

        """

        super().__init__(startN, stopN, BS)

        self.path = path
        self.device = device
        self.dtype = dtype

    def __next__(self):
        """
        Returns:
            action: (N, dim(a)), observation: (N, 3, H, W)
        """
        return _load(self.path, super().__next__(), "TN", dtype=self.dtype, device=self.device)


def _load(path, indexes, mode, dtype, device=torch.device("cpu")):
    """
    NT : shape (N, T, *)
    TN : (T, N, *)
    """

    assert mode in ("NT", "TN")

    data_dir = path
    data_action = []
    data_observation = []

    for i in indexes:
        action = (
            torch.from_numpy(np.load(Path(data_dir, f"{i}", "action.npy"))).to(dtype).to(device)
        )
        observation = (
            torch.from_numpy(np.load(Path(data_dir, f"{i}", "observation.npy")))
            .to(dtype)
            .to(device)
        )

        data_action.append(action)
        data_observation.append(observation)

    # NT
    data_action = torch.stack(data_action)
    data_observation = torch.stack(data_observation)

    if mode == "TN":
        data_observation = _swapNT(data_observation)
        data_action = _swapNT(data_action)

    return data_action, data_observation


def _swapNT(x: Tensor):
    assert x.ndim >= 3
    return x.permute((1, 0) + tuple(range(2, x.ndim)))
