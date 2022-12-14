from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch import Tensor

from mypython.ai.torch_util import swap01
from mypython.ai.util import BatchIdx


class DataLoader(BatchIdx):
    """
    Ref:
        torch.utils.data.IterableDataset
    """

    def __init__(
        self,
        root: Union[str, Path],
        start: int,
        stop: int,
        batch_size: int,
        dtype: torch.dtype,
        device=torch.device("cpu"),
    ):
        """
        root: directory path of data
            root
            ├── 0
            │   ├── action.npy
            │   ├── observation.npy
            │   ├── delta.npy
            │   └── position.npy
            ├── 1
            │   ├── action.npy
            │   ├── observation.npy
            │   ├── delta.npy
            │   └── position.npy
            ...

        """

        super().__init__(start, stop, batch_size)

        self.root = root
        self.device = device
        self.dtype = dtype

    def __next__(self):
        """
        Returns:
            action: (N, dim(u)), observation: (N, 3, H, W), ...
        """
        return _load(self.root, super().__next__(), dtype=self.dtype, device=self.device)


def _load(root, indexes, dtype, batch_first=False, device=torch.device("cpu")):
    """"""

    """
    if batch_first is True:
        (N, T, *)
    else:
        (T, N, *)
    """

    class Pack:
        def __init__(
            self, action: Tensor, observation: Tensor, delta: Tensor, position: Tensor
        ) -> None:
            self.action = action
            self.observation = observation
            self.delta = delta
            self.position = position

        def __getitem__(self, index):
            return (self.action, self.observation, self.delta, self.position)[index]

    data_dir = root

    data_action = []
    data_observation = []
    data_dt = []
    data_position = []

    def _inner_load(i, name):
        return torch.from_numpy(np.load(Path(data_dir, f"{i}", name))).to(dtype).to(device)

    for i in indexes:
        action = _inner_load(i, "action.npy")
        observation = _inner_load(i, "observation.npy")
        dt = _inner_load(i, "delta.npy")
        position = _inner_load(i, "position.npy")

        data_action.append(action)
        data_observation.append(observation)
        data_dt.append(dt)
        data_position.append(position)

    data_action = torch.stack(data_action)
    data_observation = torch.stack(data_observation)
    data_dt = torch.stack(data_dt).unsqueeze_(-1)
    data_position = torch.stack(data_position)

    if not batch_first:
        data_observation = swap01(data_observation)
        data_action = swap01(data_action)
        data_dt = swap01(data_dt)
        data_position = swap01(data_position)

    return Pack(
        action=data_action, observation=data_observation, delta=data_dt, position=data_position
    )
