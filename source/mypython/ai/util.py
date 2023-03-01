import pickle
import random
from functools import singledispatch
from pathlib import Path
from pprint import pprint
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn, optim

from mypython import rdict
from mypython.pyutil import human_readable_byte
from mypython.terminal import Color

from . import nnio

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
    if x.ndim >= 3:
        axis = (1, 0) + tuple(range(2, x.ndim))
    elif x.ndim == 2:
        axis = (1, 0)
    else:
        assert False

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


def show_model_info(model: nn.Module, verbose: bool = False):
    print("model:", model.__class__)

    _param = list(model.parameters())
    if len(_param) == 0:
        Color.print("  This model has no parameters.", c=Color.coral)
        return

    print("  dtype:", _param[0].dtype)
    print("  device:", _param[0].device)

    b_all = 0
    for name, param in model.named_parameters():
        b = param.element_size() * param.numel()
        b_all += b

        if verbose:
            if param.requires_grad:
                print(" ", name, human_readable_byte(b))
            else:
                Color.print(" ", name, human_readable_byte(b), c=Color.red)

    print("  total size:", human_readable_byte(b_all))


class BatchIdx:
    def __init__(self, start: int, stop: int, batch_size: int, shuffle=True):
        assert stop - start >= batch_size

        self._N = stop - start
        self._B = batch_size
        self._indexes = np.arange(start=start, stop=stop)
        self.shuffle = shuffle

        self._reset()

    @property
    def datasize(self):
        return self._N

    def __len__(self):
        return int(np.ceil(self._N / self._B))

    def __iter__(self):
        return self

    def __next__(self):
        self.i += 1

        if self.i == self.__len__():
            self._reset()
            raise StopIteration()

        mask = self._indexes[self.i * self._B : (self.i + 1) * self._B]
        return mask

    def _reset(self) -> None:
        if self.shuffle:
            np.random.shuffle(self._indexes)

        self.i = -1


class SequenceDataLoader(BatchIdx):
    """
    References:
        torch.utils.data.IterableDataset
    """

    def __init__(
        self,
        root: Union[str, Path],
        start: int,
        stop: int,
        batch_size: int,
        dtype: torch.dtype = torch.float32,
        device=torch.device("cpu"),
        show_selected_index=False,
        shuffle=True,
    ):
        """
        root: directory path of data
            root
            ├── 0.npz
            ├── 1.npz
            ├── 2.npz
            ...

        """

        super().__init__(start, stop, batch_size=batch_size, shuffle=shuffle)

        self.root = root
        self.device = device
        self.dtype = dtype
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
            dtype=self.dtype,
            device=self.device,
        )

    def sample_batch(self):
        batchdata = next(self)
        self._reset()
        return batchdata

    @staticmethod
    def _seq_load(
        root, indexes, dtype, batch_first=False, device=torch.device("cpu")
    ) -> Dict[str, Any]:
        """"""

        """
        if batch_first is True:
            (N, T, *)
        else:
            (T, N, *)
        """

        batch_data = {}

        for i in indexes:
            with open(Path(root, f"{i}.pickle"), "rb") as f:
                one_seq_data = pickle.load(f)
            rdict.append_a_to_b(one_seq_data, batch_data)

        rdict.to_torch(batch_data, dtype=dtype, device=device)

        if not batch_first:
            rdict.apply(batch_data, swap01)

        # rdict.show(batch_data, "batch_data (inside)")
        return batch_data
