"""
Recursive dict

Dict[str, Union[Dict[str, Tensor], Tensor]]
...
"""


import builtins
import dataclasses
from numbers import Number
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor

from .terminal import Color


def apply_(d: dict, func: Callable):
    """Inplace"""

    for k, v in d.items():
        if type(v) == dict:
            apply_(v, func)
        # ==========

        else:
            d[k] = func(v)


def apply(d: dict, func: Callable):
    d_ = {}
    _apply(d, d_, func)
    return d_


def _apply(d: dict, d_: dict, func: Callable):

    for k, v in d.items():
        if type(v) == dict:
            d_[k] = None
            _apply(v, d_[k], func)
        # ==========

        else:
            d_[k] = func(v)


# feed key chain list
# def _apply_k(d: dict, func: Callable):


# def apply_k(d: dict, func: Callable):

#     for k, v in d.items():
#         if type(v) == dict:
#             apply(v, func)
#         # ==========

#         else:
#             d[k] = func(v)


def is_scalar(scalar):
    return isinstance(scalar, Number)
    # return np.isscalar(scalar)
    # type_ = type(scalar)
    # return type_ == int or type_ == float or type_ == np.float16


def show(d: dict, name: str):
    s = f"===== show recursive dict [{name}] ====="
    print(s)
    _show(d)
    print("=" * len(s))


# def _show_iter_content(x):
#     str(x)


def _show(d: dict, start=0):
    indent = 2

    for k, v in d.items():
        k = Color.green + k + Color.reset

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

        # TODO : _show_iter_content
        elif type_v == list:
            print(" " * start + k + f": len={len(v)} (list)")
        elif type_v == tuple:
            print(" " * start + k + f": len={len(v)} (tuple)")

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
            if len(v) > 0:
                type_v0 = type(v[0])
                if type_v0 == np.ndarray:
                    d[k] = np.stack(v)
                elif type_v0 == Tensor:
                    d[k] = torch.stack(v).detach().cpu().numpy()
                else:  # float, ...
                    # print(type_v0)
                    d[k] = np.array(v)
            else:
                d[k] = np.array(v)  # empty

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
    """Elements of a are added to b (for time sequence data)"""

    # que = [a]
    # b_ = b
    # while len(que) > 0:
    #     a_ = que.pop()
    #     for ak, av in a_.items():
    #         if type(av) == dict:
    #             if not ak in b_:
    #                 b_[ak] = {}
    #             que.append(av)
    #             b_ = b_[ak]
    #             break
    #         else:
    #             if not ak in b_:
    #                 b_[ak] = [av]
    #             else:
    #                 b_[ak].append(av)

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


def either_leaf(d: dict):
    kc = []
    return _either_leaf(d, kc)


def _either_leaf(d: dict, kc: list):
    d_ = d
    while True:
        for k, v in d_.items():
            kc.append(k)
            if type(v) == dict:
                d_ = v
                break
            else:
                return v, kc


# def print_key_chain(d: dict):
#     kc = []
#     _key_chain(d, kc)


# def _key_chain(d: dict, kc: list):
#     for k, v in d.items():
#         kc.append(k)
#         if type(v) == dict:
#             _key_chain(v, kc)
#         else:
#             print(kc, v.shape)
#         kc.pop()


# @dataclasses.dataclass
# class _Wrap:
#     value = None


# def min(d: dict, comp_v: Callable):
#     min_ = _Wrap()
#     _min(d, min_, comp_v)
#     return min_.value


# def _min(d: dict, min_: _Wrap, comp_v: Callable):
#     for k, v in d.items():
#         if type(v) == dict:
#             _min(v, min_, comp_v)
#         else:
#             if min_.value is None:
#                 min_.value = comp_v(v)
#             min_.value = builtins.min(min_.value, comp_v(v))


def add_a_to_b(a: dict, b: dict):
    """Elements of a are added to b (for time sequence data)"""

    for ak, av in a.items():
        type_av = type(av)
        # print(ak, type_av)
        if type_av == dict:
            if not ak in b:
                b[ak] = {}
            add_a_to_b(av, b[ak])
        # ==========

        else:
            if not ak in b:
                b[ak] = av
            else:
                b[ak] += av
