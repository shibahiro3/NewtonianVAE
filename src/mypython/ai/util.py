import os
import pickle
import random
import sys
from functools import singledispatch
from pathlib import Path
from pprint import pprint
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from natsort import natsorted
from torch import Tensor, nn, optim
from wcmatch import glob, wcmatch

from .. import rdict
from ..pyutil import human_readable_byte
from ..terminal import Color
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


class BatchIndices:
    def __init__(self, start: int, stop: int, batch_size: int, shuffle=True):
        """
        no include stop index
        same: file[start:stop]
        """

        if not (stop - start >= batch_size):
            raise ValueError(f"Must: stop ({stop}) - start ({start}) >= batch_size ({batch_size})")

        self._N = stop - start
        self._B = batch_size
        self._indexes = np.arange(start=start, stop=stop)
        self.shuffle = shuffle

        self._reset()

    @property
    def datasize(self):
        return self._N

    @property
    def batchsize(self):
        return self._B

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


# BatchDataType = Dict[str, Any]
# BatchDataType = Any
BatchDataType = Dict[str, Union[Tensor, Dict[str, Tensor]]]


def file_collector(patterns: Union[Union[str, Path], List[Union[str, Path]]]):
    """
    https://facelessuser.github.io/wcmatch/glob/
    """

    def _fc(ptn: str):
        return glob.glob(patterns=ptn, flags=wcmatch.GLOBSTAR | wcmatch.BRACE)
        # return Path().glob(ptn)

    filelist = []
    if type(patterns) == list:
        for p in patterns:
            fs = _fc(p)
            filelist += fs
    else:
        filelist = _fc(patterns)

    filelist = list(set(filelist))
    filelist = natsorted(filelist)

    return filelist


class SequenceDataLoader(BatchIndices):
    """
    References:
        torch.utils.data.IterableDataset
    """

    def __init__(
        self,
        patterns: Union[Union[str, Path], List[Union[str, Path]]],
        batch_size: Optional[int] = None,
        max_time: Union[int, str] = "clip_min",
        dtype: Union[str, torch.dtype, None] = None,
        device: Union[str, torch.device, None] = None,
        show_selected_index=False,
        shuffle=True,
        preprocess: Optional[Callable[[BatchDataType], BatchDataType]] = None,
    ):
        """

        max_time : int or str
            int : max_time
            str :
                "clip_min" : Truncate from batch data according to minimum time length
        """

        self._filelist = file_collector(patterns)

        if len(self._filelist) == 0:
            raise Exception("Data file doesn't exist")

        stop = len(self._filelist)
        if batch_size is None:
            batch_size = stop

        super().__init__(0, stop, batch_size=batch_size, shuffle=shuffle)

        if type(dtype) == str:
            dtype = getattr(torch, dtype)
        if type(device) == str:
            device = torch.device(device)

        # self.root = root
        self.device = device
        self.dtype = dtype
        self.max_time = max_time
        self.show_selected_indices = show_selected_index
        self.preprocess = preprocess
        # self.flist = list(Path(self.root).glob("*"))

    def __next__(self) -> BatchDataType:
        """
        Returns:
            batch_data : Dict[str, ...]  ((T, B, *) Without preprocess)
        """

        indices = super().__next__()

        if self.show_selected_indices:
            msg = f"=== Selected indices ({self.__class__.__name__}) ==="
            print(msg)
            print(indices)
            print("=" * len(msg))

        ### load and clip ###
        batch_data = {}
        min_t = None
        for i in indices:
            # one_seq_data = self._pickle_load(Path(self.root, f"{i}.pickle"))  # (T, *)
            one_seq_data = self._pickle_load(self._filelist[i])  # (T, *)

            ### for "clip_min" ###
            leaf, kc = rdict.either_leaf(one_seq_data)
            if min_t is None:
                min_t = len(leaf)
            else:
                min_t = min(min_t, len(leaf))
            ###

            if type(self.max_time) == int:
                rdict.apply_(one_seq_data, lambda x: x[: self.max_time])
            rdict.append_a_to_b(one_seq_data, batch_data)  # to (B, [T, *])

        if self.max_time == "clip_min":
            rdict.apply_(batch_data, lambda x: [one_seq[:min_t] for one_seq in x])
        rdict.to_torch(batch_data, dtype=self.dtype)  # to (B, T, *)
        #################

        ### transform ###
        # TODO: rdict.pad_time(...
        rdict.apply_(batch_data, swap01)  # to: (T, B, *)
        if self.preprocess is not None:
            batch_data = self.preprocess(batch_data)
        #################

        rdict.to_torch(batch_data, dtype=self.dtype, device=self.device)
        # rdict.apply_(batch_data, lambda x: x.detach())  # Unnecessary
        return batch_data

    def sample_batch(self, batch_size: Union[int, str] = "same", verbose=False, print_name=None):
        batch_size_prev = self.batchsize
        if type(batch_size) == str:
            assert batch_size in ("same", "all")
            if batch_size == "all":
                self._B = self.datasize

        batchdata = next(self)
        self._reset()
        if verbose:
            if print_name is None:
                print_name = "sample batch data"
            rdict.show(batchdata, print_name)
            print(f"Data size: {self.datasize}")
            print(f"Batch size: {self.batchsize}")
            print(f"Iterations size of per epoch: {len(self)}")

        self._B = batch_size_prev
        return batchdata

    @staticmethod
    def _pickle_load(p):
        with open(p, "rb") as f:
            ret = pickle.load(f)
        return ret


_T = TypeVar("_T")


def random_sample(x: _T) -> _T:
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    return x[indices]
