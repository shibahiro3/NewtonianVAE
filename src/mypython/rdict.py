"""
Recursive dict

Dict[str, Union[Dict[str, Tensor], Tensor]]
...
"""


import builtins
import dataclasses
import pickle
from numbers import Number
from typing import Any, Callable, List, Optional, Union

import numpy as np
import torch
from torch import Tensor

from mypython.pyutil import human_readable_byte

from .terminal import Color


def get(d: dict, key_chain: list, default=None):
    """like nullsafe"""

    class _KeyError:
        pass

    tmp = d
    for k in key_chain:
        if type(tmp) != dict:
            return default
        tmp = tmp.get(k, _KeyError())
        # tmpをtype()で縛らないと、tmpがnumpyのとき==がelement is ambiguous. Use a.any() or a.all()言うてくる
        if type(tmp) == _KeyError:
            return default
    return tmp


def apply(d: dict, func: Callable):
    d_ = {}
    _apply(d, d_, func)
    return d_


def apply_(d: dict, func: Callable):
    """Inplace"""

    for k, v in d.items():
        if type(v) == dict:
            apply_(v, func)
        # ==========

        else:
            d[k] = func(v)

    return d


def _apply(d: dict, d_: dict, func: Callable):

    for k, v in d.items():
        if type(v) == dict:
            d_[k] = {}
            _apply(v, d_[k], func)
        # ==========

        else:
            d_[k] = func(v)


def apply_kc_(d: dict, func: Callable[[list, Any], None]):
    """func(key_chain, value)"""
    _apply_kc_(d, [], func)


def _apply_kc_(d: dict, parent_keys: list, func: Callable[[list, Any], None]):
    for k, v in d.items():
        key_chain = parent_keys + [k]
        if isinstance(v, dict):
            _apply_kc_(v, key_chain, func)
        else:
            func(key_chain, v)


def print_keys(d: dict):
    apply_kc_(d, lambda key_chain, v: print(key_chain))


def transfer_from_kp(dsrc: dict, ddst: dict, keypath: list):
    """keypathと一致するdsrcをddstに転送する"""
    _transfer_from_kp(dsrc, ddst, keypath, 0)


def _transfer_from_kp(dsrc: dict, ddst: dict, keypath: list, i: int):
    if keypath[i] in dsrc:
        if i < len(keypath) - 1:
            if not keypath[i] in ddst:
                ddst[keypath[i]] = {}
            _transfer_from_kp(dsrc[keypath[i]], ddst[keypath[i]], keypath, i + 1)
        else:
            ddst[keypath[i]] = dsrc[keypath[i]]


def extract_from_keypaths(d: dict, keypaths: Optional[List[list]] = None):
    """
    Ex.

    d = {
        "key1": random.randint(0, 127),
        "key2": {
            "key2.1": random.randint(0, 127),
            "key2.2": random.randint(0, 127),
            "key2.3_NOT_NEED": random.randint(0, 127),
        },
        "key3_NOT_NEED": random.randint(0, 127),
        "key4": {
            "key4.1_NOT_NEED": random.randint(0, 127),
            "key4.2": random.randint(0, 127),
        },
        "key5": random.randint(0, 127),
        "key6": {},
        "key7": {
            "key7.3": {
                "key7.3.5": random.randint(0, 127),
                "key7.3.X_NOT_NEED": random.randint(0, 127),
            },
            "key7.4_NOT_NEED": random.randint(0, 127),
        },
        "key8": [],
        "key9": random.randint(0, 127),
        123: {"456": {789: "DATA"}},
    }

    keypaths = [
        ["key1"],
        ["key2", "key2.1"],
        ["key2", "key2.2"],
        ["key4"],  # -> all
        ["key6"],
        ["key7", "key7.3", "key7.3.5"],
        ["key100"],  # ignore
        "key9", # if only top, not necessary list
        [123, "456", 789],
    ]

    rdict.show(d)
    extracted = rdict.extract_from_keypaths(d, keypaths)
    rdict.show(extracted)

    """
    if keypaths is None:
        return d
    else:
        extracted = {}
        for kpath in keypaths:
            if type(kpath) != list:
                kpath = [kpath]
            transfer_from_kp(d, extracted, kpath)
        return extracted


def remove_from_keypaths(d: dict, keypaths: Optional[List[list]] = None):
    pass


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


class SwitchPrint:
    def __init__(self, enable=True) -> None:
        self.enable = enable

    def __call__(self, *args, **kwargs) -> Any:
        if self.enable:
            builtins.print(*args, **kwargs)


def show(d: dict, name: Optional[str] = None, print_precision: int = 2, bin=True, only_info=False):
    # bin == True : *iB (nvidia-smi)

    print = SwitchPrint(not only_info)

    if name is None:
        s = f"========== show recursive dict =========="
    else:
        s = f"========== show recursive dict [{name}] =========="
    print(s)
    nbytes = [0]
    _show(d, print_precision, 0, nbytes, bin, print)
    print("Total array size:", human_readable_byte(nbytes[0], bin=bin))
    print("=" * len(s))

    info = dict(nbytes=nbytes[0])

    return info


# def _show_iter_content(x):
#     str(x)


# def _isArrayLike(obj):
#     # O: str
#     return hasattr(obj, "__iter__") and hasattr(obj, "__len__")


def _show(d: dict, print_precision, start, nbytes, bin, print):
    indent = 2

    for k, v in d.items():
        k = Color.green + str(k) + Color.reset

        if type(v) == dict:
            if not v:
                print(" " * start + f"{k}: (empty dict)")
            else:
                print(" " * start + f"{k}:")
                _show(v, print_precision, start + indent, nbytes, bin, print)
        # ==========

        elif type(v) == np.ndarray:
            nb = v.nbytes
            nbytes[0] += nb
            if np.issubdtype(v.dtype, np.number):
                print(
                    " " * start
                    + k
                    + f": shape={tuple(v.shape)} "
                    + range_str(v, print_precision)
                    + f" (numpy, {v.dtype}, {human_readable_byte(nb, bin=bin)})"
                )
            else:
                print(" " * start + k + f": shape={tuple(v.shape)} (numpy, {v.dtype})")
        elif type(v) == Tensor:
            nb = v.element_size() * v.numel()
            nbytes[0] += nb
            dtyp = str(v.dtype).split(".")[1]
            print(
                " " * start
                + k
                + f": shape={tuple(v.shape)} "
                + range_str(v, print_precision)
                + f" (torch, {dtyp}, {human_readable_byte(nb, bin=bin)}, {v.device})"
            )

        elif type(v) == list or type(v) == tuple or type(v) == set:
            v_str = str(v)
            N_ = 25
            if len(v_str) <= N_:
                print(" " * start + k + f": {v_str[:N_]} ({type(v)}, len={len(v)})")
            else:
                print(" " * start + k + f": {v_str[:N_]}... ({type(v)}, len={len(v)})")

        else:
            print(" " * start + k + f": {v} ({type(v).__name__})")


def range_str(x, print_precision):
    min = x.min()  # .item()
    max = x.max()  # .item()
    if (type(x) == np.ndarray and np.issubdtype(x.dtype, np.floating)) or (
        type(x) == Tensor and torch.is_floating_point(x)
    ):
        return f"range=[{min:.{print_precision}f}, {max:.{print_precision}f}]"
    else:
        return f"range=[{min}, {max}]"


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


def to_torch(d: dict, *, ignore_scalar=False, dtype=None, device=None, contiguous=None):
    # cpuでcontiguousであってもto(cuda)では引き継がれない
    # それに大抵がbatch selectのせいで事前のcontiguousをしても意味がない

    for k, v in d.items():
        type_v = type(v)
        # is_torch = True
        # Color.print(k, type_v)
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

        # 再帰で戻ってきたとき、torch であるとは限らない (まだdictの可能性あり)
        # print(k, type(d[k]))
        d[k] = _torch_to(d[k], dtype=dtype, device=device, contiguous=contiguous)


def _torch_to(x, dtype=None, device=None, contiguous=None):
    if type(x) == Tensor:
        if dtype:
            x = x.to(dtype)
        if device:
            x = x.to(device)
        if contiguous:
            x = x.contiguous()
    return x


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


def save(d: dict, path):
    with open(path, "wb") as f:
        pickle.dump(d, f)


# def save_npy(d: dict):
#     pass


def load(path):
    with open(path, "rb") as f:
        ret = pickle.load(f)
    return ret


# def load_npy(path, keypaths=None):
#     pass
