import random
from pathlib import Path
from pprint import pprint
from typing import Callable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn, optim

_NT = Union[np.ndarray, Tensor]


def reproduce(seed: int = 1234) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_module_params(module: nn.Module, grad=False) -> None:
    print("=== Model Parameters ===")
    print(type(module))
    for name, param in module.named_parameters(recurse=True):
        print(f"name: {name} | shape: {[*param.shape]}")
        if grad:
            if type(param.grad) == Tensor:
                print(f"grad: \n{param.grad}\n")
            else:
                print(f"grad: {param.grad}")  # None


def swish(x):
    return x * torch.sigmoid(x)


def mish(x):
    return x * torch.tanh(F.softplus(x))


def find_function(function_name: str) -> Callable[[Tensor], Tensor]:
    try:
        return getattr(torch, function_name)
    except:
        return getattr(F, function_name)


def swap01(x: _NT) -> _NT:
    # assert x.ndim >= 3
    axis = (1, 0) + tuple(range(2, x.ndim))
    if type(x) == np.ndarray:
        return x.transpose(axis)
    elif type(x) == Tensor:
        return x.permute(axis)
    else:
        assert False


def to_np(x) -> np.ndarray:
    _type = type(x)
    if _type == Tensor:
        return x.detach().cpu().numpy()
    elif _type == np.ndarray:
        return x
    else:
        return np.array(x)


class BatchIdx:
    def __init__(self, start: int, stop: int, batch_size: int):
        assert stop - start >= batch_size

        self.N = stop - start
        self.BS = batch_size
        self.indexes = np.arange(start=start, stop=stop)
        self._reset()

    def __len__(self):
        return int(np.ceil(self.N / self.BS))

    def __iter__(self):
        return self

    def __next__(self):
        self.i += 1

        if self.i == self.__len__():
            self._reset()
            raise StopIteration()

        mask = self.indexes[self.i * self.BS : (self.i + 1) * self.BS]
        return mask

    def _reset(self):
        np.random.shuffle(self.indexes)
        self.i = -1


class SequenceDataLoader(BatchIdx):
    """
    References:
        torch.utils.data.IterableDataset
    """

    def __init__(
        self,
        root: Union[str, Path],
        names: list,
        start: int,
        stop: int,
        batch_size: int,
        dtype: torch.dtype,
        device=torch.device("cpu"),
        show_selected_index=False,
    ):
        """
        root: directory path of data
            root
            ├── 0
            │   ├── names[0].npy
            │   ├── names[1].npy
            │   ├── names[2].npy
            │   ...
            ├── 1
            │   ├── names[0].npy
            │   ├── names[1].npy
            │   ├── names[2].npy
            │   ...
            ...

        """

        super().__init__(start, stop, batch_size)

        self.root = root
        self.device = device
        self.dtype = dtype
        self.names = names
        self.show_selected_index = show_selected_index

    def __next__(self):
        mask = super().__next__()

        if self.show_selected_index:
            msg = f"=== Selected index ({self.__class__.__name__}) ==="
            print(msg)
            print(mask)
            print("=" * len(msg))

        return self._seq_load(
            root=self.root,
            indexes=mask,
            names=self.names,
            dtype=self.dtype,
            device=self.device,
        )

    @staticmethod
    def _seq_load(
        root, indexes, dtype, names: list, batch_first=False, device=torch.device("cpu")
    ) -> List[Tensor]:
        """"""

        """
        if batch_first is True:
            (N, T, *)
        else:
            (T, N, *)
        """

        batch_data: List[List[Tensor]] = []
        for _ in range(len(names)):
            batch_data.append([])

        def _inner_load(i, name):
            return torch.from_numpy(np.load(Path(root, f"{i}", name))).to(dtype).to(device)

        for i in indexes:
            for j in range(len(names)):
                batch_data[j].append(_inner_load(i, names[j] + ".npy"))

        for j in range(len(names)):
            batch_data[j] = torch.stack(batch_data[j])

        if not batch_first:
            for j in range(len(names)):
                batch_data[j] = swap01(batch_data[j])

        return batch_data
