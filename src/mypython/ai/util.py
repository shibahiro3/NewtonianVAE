import gc
import pickle
import random
import sys
from pathlib import Path
from pprint import pprint
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np
import torch
import torch.nn.functional as F
from natsort import natsorted
from torch import Tensor, nn
from torchvision.io import read_image
from wcmatch import glob, wcmatch

from .. import rdict
from ..pyutil import human_readable_byte
from ..terminal import Color

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


def to_numpy(x) -> np.ndarray:
    if type(x) == Tensor:
        return x.detach().cpu().numpy()
    elif type(x) == np.ndarray:
        return x
    else:
        return np.array(x)


def to_torch(x) -> Tensor:
    def _has_neg(x):
        for element in x:
            if element < 0:
                return True
        return False

    if type(x) == np.ndarray:
        if _has_neg(x.strides):
            return torch.from_numpy(x.copy())
        else:
            return torch.from_numpy(x)
    elif type(x) == Tensor:
        return x
    else:
        return torch.tensor(x)


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

    print("  total size:", human_readable_byte(b_all, bin=True))


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
        self._indices = np.arange(start=start, stop=stop)
        self.shuffle = shuffle
        self.reset_indices()

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
        self._i += 1

        if self._i == self.__len__():
            self.reset_indices()
            raise StopIteration()

        indices = self._indices[self._i * self._B : (self._i + 1) * self._B]
        return indices

    def reset_indices(self) -> None:
        if self.shuffle:
            np.random.shuffle(self._indices)
        self._i = -1


# BatchDataType = Dict[str, Any]
# BatchDataType = Any
BatchDataType = Dict[str, Union[Tensor, Dict[str, Tensor]]]


def file_collector(patterns: Union[Union[str, Path], List[Union[str, Path]]]):
    """
    https://facelessuser.github.io/wcmatch/glob/
    """

    filelist = glob.glob(patterns=patterns, flags=wcmatch.GLOBSTAR | wcmatch.BRACE, limit=0)
    # filelist = list(set(filelist)) # already (input list)
    filelist = natsorted(filelist)
    return filelist


class SequenceDataLoader(BatchIndices):
    """
    HACK:
    https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
    https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
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
        batch_first=False,
        preprocess: Optional[Callable[[BatchDataType], BatchDataType]] = None,
        keypaths=None,
        load_all=False,
    ):
        """
        max_time : int or str
            int : max_time
            str :
                "clip_min" : Truncate from batch data according to minimum time length
        """

        # 色々並列化試した(並列load, preload)がほどんど同じ、逆効果だった

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

        self.device = device
        self.dtype = dtype
        self.max_time = max_time
        self.show_selected_indices = show_selected_index
        self.batch_first = batch_first
        self.preprocess = preprocess
        self.keypaths = keypaths

        self.all_data = None
        if load_all:
            try:
                self.all_data = self._load(self._filelist)
                nbytes = rdict.show(self.all_data, only_info=True)["nbytes"]
                Color.print(
                    f"Loaded all data size: {human_readable_byte(nbytes, bin=True)}",
                    c=Color.green,
                )
            except:
                gc.collect()
                Color.print("WARNING: Failed load all", c=Color.coral)

    def sample_batch(self, batch_size: Union[int, str] = "same", verbose=False, print_name=None):
        # TODO: each sequence List[dict]  for large batch_size for correlation  (torch.cuda.OutOfMemoryError)

        batch_size_prev = self.batchsize
        if type(batch_size) == str:
            assert batch_size in ("same", "all")
            if batch_size == "all":
                self._B = self.datasize

        batchdata = next(self)
        self.reset_indices()
        if verbose:
            if print_name is None:
                print_name = "sample batch data"
            rdict.show(batchdata, print_name)
            print(f"Data size: {self.datasize}")
            print(f"Batch size: {self.batchsize}")
            print(f"Iterations of per epoch: {len(self)}")
            print("-" * 35)

        self._B = batch_size_prev
        return batchdata

    def __next__(self) -> BatchDataType:
        """
        Returns:
            batch_data : Dict[str, ...]  (T, B, *)
        """

        indices = super().__next__()
        if self.show_selected_indices:
            msg = f"=== Selected indices ({self.__class__.__name__}) ==="
            print(msg)
            print(indices)
            print("=" * len(msg))

        if self.all_data is None:
            batch_data = self._load((self._filelist[i] for i in indices))
        else:
            batch_data = rdict.apply(self.all_data, lambda x: x[indices])

        batch_data = self._preprocess(batch_data)
        return batch_data

    def _preprocess(self, batch_data):
        # CPU -> GPU -> preprocess -> for training data

        rdict.to_torch(batch_data, device=self.device)
        # TODO: rdict.pad_time(...
        if not self.batch_first:
            rdict.apply_(batch_data, swap01)  # to (T, B, *)
        if self.preprocess is not None:
            batch_data = self.preprocess(batch_data)
        rdict.to_torch(batch_data, dtype=self.dtype, device=self.device)  # for dtype
        return batch_data

    def _load(self, filenames):
        # Only on CPU

        batch_data = {}
        min_t = None
        for fname in filenames:
            one_seq_data = _pickle_load(fname)  # (T, *)
            if self.keypaths is not None:  # save memory ... ?  nvidia-smi result is same
                one_seq_data = rdict.extract_from_keypaths(one_seq_data, self.keypaths)

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
        rdict.to_torch(batch_data)  # to (B, T, *), UNCHANGED dtype (for uint8)
        return batch_data


def _pickle_load(p):
    with open(p, "rb") as f:
        ret = pickle.load(f)
    return ret


_T = TypeVar("_T")


def random_sample(x: _T, k: int) -> _T:
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    return x[indices[:k]]