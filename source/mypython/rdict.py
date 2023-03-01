"""
Recursive dict

Dict[str, Union[Dict[str, Tensor], Tensor]]
...
"""

from numbers import Number
from typing import Callable

import numpy as np
import torch
from torch import Tensor


def apply(d: dict, func: Callable):
    for k, v in d.items():
        if type(v) == dict:
            apply(v, func)
        # ==========

        else:
            d[k] = func(v)


def is_scalar(scalar):
    return isinstance(scalar, Number)
    # return np.isscalar(scalar)
    # type_ = type(scalar)
    # return type_ == int or type_ == float or type_ == np.float16


def show(d: dict, name: str):
    print(f"===== show dict [{name}] =====")
    _show(d)


def _show(d: dict, start=0):
    indent = 2

    for k, v in d.items():
        type_v = type(v)
        if type_v == dict:
            print(" " * start + f"{k}:")
            _show(v, start + indent)
        # ==========

        elif type_v == np.ndarray:
            v: np.ndarray
            print(" " * start + k + f": shape={tuple(v.shape)} (numpy, {v.dtype})")
        elif type_v == Tensor:
            v: Tensor
            print(" " * start + k + f": shape={tuple(v.shape)} ({v.dtype}, {v.device})")
        elif type_v == list:
            print(" " * start + k + f": len={len(v)} (list)")
        else:
            print(" " * start + k + f": {v} ({type_v.__name__})")


def to_numpy(d: dict, *, ignore_scalar=False):
    for k, v in d.items():
        type_v = type(v)
        if type_v == dict:
            to_numpy(v, ignore_scalar=ignore_scalar)
        # ==========

        elif type_v == np.ndarray:
            pass

        elif type_v == Tensor:
            v: Tensor
            d[k] = v.detach().cpu().numpy()

        elif type_v == list:
            type_v0 = type(v[0])
            if type_v0 == np.ndarray:
                d[k] = np.stack(v)
            elif type_v0 == Tensor:
                d[k] = torch.stack(v).detach().cpu().numpy()
            else:  # float, ...
                d[k] = np.array(v)

        elif is_scalar(v):
            if not ignore_scalar:
                d[k] = np.array(v)

        else:
            raise TypeError(f'type of "{k}" is {type_v.__name__}')


def _to_dtype_device(x, dtype=None, device=None):
    if type(x) == Tensor:
        if dtype is not None:
            x = x.to(dtype)
        if device is not None:
            x = x.to(device)
    return x


def to_torch(d: dict, *, ignore_scalar=False, dtype=None, device=None):
    for k, v in d.items():
        type_v = type(v)
        # print(k, type_v)
        if type_v == dict:
            to_torch(v, ignore_scalar=ignore_scalar, dtype=dtype, device=device)
        # ==========

        elif type_v == np.ndarray:
            d[k] = torch.from_numpy(v)

        elif type_v == Tensor:
            pass

        elif type_v == list:
            type_v0 = type(v[0])
            if type_v0 == np.ndarray:
                d[k] = torch.from_numpy(np.stack(v))
            elif type_v0 == Tensor:
                d[k] = torch.stack(v)
            else:  # float, ...
                d[k] = torch.tensor(v)

        elif is_scalar(v):
            if not ignore_scalar:
                d[k] = torch.tensor(v)

        else:
            raise TypeError(f'type of "{k}" is {type_v.__name__}')

        d[k] = _to_dtype_device(d[k], dtype=dtype, device=device)


def append_a_to_b(a: dict, b: dict):
    """Elements of a are added to b"""

    for ak, av in a.items():
        type_av = type(av)
        # print(ak, type_av)
        if type_av == dict:
            if not ak in b:
                b[ak] = {}
            append_a_to_b(av, b[ak])
        # ==========

        else:
            if not ak in b:
                b[ak] = [av]
            else:
                b[ak].append(av)
