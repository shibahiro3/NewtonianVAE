"""
Recursive dict
Hierarchical Structure
For data, config, etc.

Ex:
    Dict[str, Union[Dict[str, Tensor], Tensor]]
"""


import builtins
import collections
import dataclasses
import pickle
import pprint as builtins_pprint
from numbers import Number
from pprint import pformat
from typing import Any, Callable, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from torch import Tensor

from mypython.pyutil import human_readable_byte

from .terminal import Color

D = builtins.dict
# collections.OrderedDict
# collections.UserDict


def is_dict(v):
    return isinstance(v, dict)
    # return hasattr(v, "items")
    # return type(v) == dict


def is_leaf(v):
    return not is_dict(v)


def dfs_loop(
    d: D, f: Optional[Callable[[dict, list, Any], None]] = None, list_type: str = "none"
) -> List[Tuple[list, Any]]:
    """
    f(dict_parent, keylist, value)
    f(dp, kl, v)
        value access: dp[kl[-1]] = ...
        dp[kl[-1]] and value are the same

        leaf access:
            if not is_dict(v):
                dp[kl[-1]] = ...
    """

    assert list_type in ("none", "raw", "applied")

    stack = [(d, [], d)]
    record = []

    while stack:
        v, kl, dp = stack.pop()

        if (len(kl) > 0) and (f is not None):
            f(dp, kl, v)

        if is_dict(v):  # is not leaf
            for kk, vv in reversed(v.items()):  # stackだから逆順にすると、逆の逆で もとのdict通りの順序になる
                stack.append((vv, kl + [kk], v))
        elif list_type != "none":  # record leaf
            if list_type == "raw":
                record.append((kl, v))
            elif list_type == "applied":
                record.append((kl, dp[kl[-1]]))

    return record


def get(d: D, keylist: Union[list, str], default=None, sep="/"):
    """Like nullsafe"""

    class _KeyError:
        pass

    c = d
    for k in _proper_keylist(keylist, sep):
        c = c.get(k, _KeyError())
        # tmpをtype()で縛らないと、tmpがnumpyのとき==がelement is ambiguous. Use a.any() or a.all()言うてくる
        if type(c) == _KeyError:
            return default
    return c


def set(d: D, keylist: Union[list, str], value, sep="/"):
    _set(d, 0, _proper_keylist(keylist, sep), value)


def _set(d: D, i: int, kl: list, value):
    if i == len(kl) - 1:
        d[kl[i]] = value
    else:
        k = kl[i]
        if is_dict(d):
            if k not in d:
                d[k] = {}
        if not is_dict(d[k]):
            d[k] = {}
        _set(d[k], i + 1, kl, value)


def _proper_keylist(keylist: Union[list, str], sep="/"):
    if type(keylist) == str:
        return [k for k in keylist.split(sep) if k != ""]

    return keylist


def apply(d: D, func: Callable):
    d_ = {}
    _apply(d, d_, func)
    return d_


def apply_(d: D, func: Callable):
    """Inplace"""

    for k, v in d.items():
        if is_dict(v):
            apply_(v, func)
        else:
            d[k] = func(v)
    return d


def _apply(d: D, d_: D, func: Callable):
    for k, v in d.items():
        if is_dict(v):
            d_[k] = {}
            _apply(v, d_[k], func)
        else:
            d_[k] = func(v)


def apply_kc_(d: D, func: Callable[[list, Any], None]):
    """func(key_chain, value)"""
    _apply_kc_(d, [], func)


def _apply_kc_(d: D, parent_keys: list, func: Callable[[list, Any], None]):
    for k, v in d.items():
        key_chain = parent_keys + [k]
        if is_dict(v):
            _apply_kc_(v, key_chain, func)
        else:
            func(key_chain, v)


def print_keys(d: D):
    apply_kc_(d, lambda key_chain, v: builtins.print(key_chain))


def _transfer_from_kp(dsrc: D, ddst: D, keypath: list):
    """keypathと一致するdsrcをddstに転送する"""
    _transfer_from_kp_(dsrc, ddst, keypath, 0)


def _transfer_from_kp_(dsrc: D, ddst: D, keypath: list, i: int):
    if keypath[i] in dsrc:
        if i < len(keypath) - 1:
            if not keypath[i] in ddst:
                ddst[keypath[i]] = {}
            _transfer_from_kp_(dsrc[keypath[i]], ddst[keypath[i]], keypath, i + 1)
        else:
            ddst[keypath[i]] = dsrc[keypath[i]]


def extract_from_keypaths(d: D, keypaths: Optional[List[list]] = None):
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
            _transfer_from_kp(d, extracted, kpath)
        return extracted


def remove_from_keypaths(d: D, keypaths: Optional[List[list]] = None):
    pass


def is_scalar(scalar):
    return isinstance(scalar, Number)
    # return np.isscalar(scalar)
    # type_ = type(scalar)
    # return type_ == int or type_ == float or type_ == np.float16


# class SwitchPrint:
#     def __init__(self, enable=True) -> None:
#         self.enable = enable

#     def __call__(self, *args, **kwargs) -> Any:
#         if self.enable:
#             builtins.print(*args, **kwargs)


class _Mem:
    def __init__(self, value=None) -> None:
        self.value = value


def _pprint_v(v, precision: int, bin: bool, str_lengths: int = 32):
    # pl.Config.set_fmt_str_lengths

    nbyte = 0

    if is_dict(v):
        if not v:
            S = "(empty dict)"
        else:
            S = ""

    elif type(v) == np.ndarray:
        nbyte = v.nbytes
        if np.issubdtype(v.dtype, np.number):
            S = (
                f"shape={tuple(v.shape)} "
                + range_str(v, precision)
                + f" (numpy, {v.dtype}, {human_readable_byte(nbyte, bin=bin)})"
            )
        else:
            S = f"shape={tuple(v.shape)} (numpy, {v.dtype})"

    elif type(v) == Tensor:
        nbyte = v.element_size() * v.numel()
        dtyp = str(v.dtype).split(".")[1]
        S = (
            f"shape={tuple(v.shape)} "
            + range_str(v, precision)
            + f" (torch, {dtyp}, {human_readable_byte(nbyte, bin=bin)}, {v.device})"
        )

    elif type(v) == list or type(v) == tuple or type(v) == builtins.set:
        v_str = str(v)
        # v_str = pformat(v)  # TODO: \n があったらどうする?

        if len(v_str) <= str_lengths:
            S = f"{v_str} ({type(v).__name__}, len={len(v)})"
        else:
            S = f"{v_str[:str_lengths]} ... {v_str[-5:-1]} ({type(v).__name__}, len={len(v)})"

    else:
        # TODO: pprint.pformat(v)
        S = f"{v} ({type(v).__name__})"

    return S, nbyte


def pprint(
    d: D,
    name: Optional[str] = None,
    precision: int = 2,
    bin: bool = True,
    align: bool = False,
    key_color: str = Color.green,
    bg_color: str = "",  # Color.bg_rgb(40, 25, 40)
):
    """pretty-print

    Args:
        bin: if True : *iB (same as nvidia-smi)

    Returns:
        info
    """

    if name is None:
        s = f"========== show recursive dict =========="
    else:
        s = f"========== show recursive dict [{name}] =========="
    builtins.print(s)

    nbytes = _Mem(0)
    even_odd = _Mem(True)
    indent = 2
    m = _max_namelabel_len(d, indent)

    def inner(dp, kl, v):
        k = kl[-1]
        S, nbyte = _pprint_v(v, precision, bin)
        nbytes.value += nbyte

        even_odd.value = not even_odd.value
        if even_odd.value:
            bg = bg_color
        else:
            bg = ""

        k = str(k)
        depth = len(kl) - 1

        # -----
        space0 = "  " * depth

        # -- or --

        # if depth == 1:
        #     ...

        # space0 = "├─" * depth
        # -----

        if align:
            space1 = " " * (m - depth * indent - len(k))
        else:
            space1 = ""

        builtins.print(
            bg + space0 + key_color + k + Color.reset + bg + ": " + space1 + S + Color.reset
        )

    dfs_loop(d, inner)
    info = dict(nbytes=nbytes.value)
    return info


def to_list(d: D):
    return dfs_loop(d, list_type="raw")


# def _dfs_recur(d: D):  # 不要になった..
#     pass


# def _dfs_loop_show(d: D):  # test
#     record = dfs_loop(d, lambda dc, kc, v: print(kc, dc[kc[-1]]))

#     def f(dp, kl, v):
#         is_same = dp[kl[-1]] == v
#         if type(v) == np.ndarray or type(v) == torch.Tensor:
#             assert is_same.all()
#         else:
#             assert is_same

#         dp[kl[-1]] = "AAAAA"  # write

#     record = dfs_loop(d, f, list_type=True)
#     builtins_pprint.pprint(record)

#     record = dfs_loop(
#         d,
#         lambda dc, kc, v: print(Color.green + "/".join(map(str, kc)) + Color.reset + " :", v),
#     )
#     builtins_pprint.pprint(record)


def _max_namelabel_len(d: D, indent: int):
    stack = [(d, 0, 0)]
    m = 0
    while stack:
        d_c, l, dep_c = stack.pop()  # current
        m = max(m, l)
        for k, v in reversed(d_c.items()):
            if is_dict(v):
                stack.append((v, indent * dep_c, dep_c + 1))
            else:
                stack.append(({}, indent * dep_c + len(str(k)), dep_c + 1))
    return m


# def _isArrayLike(obj):
#     # O: str
#     return hasattr(obj, "__iter__") and hasattr(obj, "__len__")


def range_str(x, precision):
    min = x.min()  # .item()
    max = x.max()  # .item()
    if (type(x) == np.ndarray and np.issubdtype(x.dtype, np.floating)) or (
        type(x) == Tensor and torch.is_floating_point(x)
    ):
        return f"range=[{min:.{precision}f}, {max:.{precision}f}]"
    else:
        return f"range=[{min}, {max}]"


def to_numpy_(d: D, *, ignore_scalar=False):
    for k, v in d.items():
        type_v = type(v)
        if is_dict(v):
            to_numpy_(v, ignore_scalar=ignore_scalar)

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

    return d


def to_torch_(d: D, *, ignore_scalar=False, dtype=None, device=None, contiguous=None):
    # cpuでcontiguousであってもto(cuda)では引き継がれない
    # それに大抵がbatch selectのせいで事前のcontiguousをしても意味がない

    for k, v in d.items():
        type_v = type(v)
        # is_torch = True
        # Color.print(k, type_v)
        if is_dict(v):
            to_torch_(v, ignore_scalar=ignore_scalar, dtype=dtype, device=device)

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

    return d


def _torch_to(x, dtype=None, device=None, contiguous=None):
    if type(x) == Tensor:
        x = x.to(dtype=dtype, device=device)
        if contiguous:
            x = x.contiguous()
    return x


def append_a_to_b(a: D, b: D):
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
        # type_av = type(av)
        # print(ak, type_av)
        if is_dict(av):
            if not ak in b:
                b[ak] = {}
            append_a_to_b(av, b[ak])
        else:
            if not ak in b:
                b[ak] = [av]
            else:
                b[ak].append(av)


def add_a_to_b(a: D, b: D):
    """Elements of a are added to b (for time sequence data)"""

    for ak, av in a.items():
        # type_av = type(av)
        # print(ak, type_av)
        if is_dict(av):
            if not ak in b:
                b[ak] = {}
            add_a_to_b(av, b[ak])
        else:
            if not ak in b:
                b[ak] = av
            else:
                b[ak] += av


def either_leaf(d: D):
    kc = []
    return _either_leaf(d, kc)


def _either_leaf(d: D, kc: D):
    d_ = d
    while True:
        for k, v in d_.items():
            kc.append(k)
            if is_dict(v):
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


def save(d: D, path):
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

### h5py ###


def pprint_hdf5(root: h5py.Group):
    s = "=============== HDF5 ==============="
    print(s)
    # print(root.id)
    # print(root.file)
    attrs = dict(root.attrs)
    if attrs:
        print("root:")
        print("  attrs:", attrs)

    def range_str(x, precision):
        min = np.min(x)
        max = np.max(x)
        if np.issubdtype(x.dtype, np.floating):
            return f"range=[{min:.{precision}f}, {max:.{precision}f}]"
        else:
            return f"range=[{min}, {max}]"

    def PrintOnlyDataset(name, obj: Union[h5py.Group, h5py.Dataset]):
        # assert "/" + name == obj.name
        name = obj.name
        key_color = Color.green
        precision = 2
        attrs = dict(obj.attrs)
        if isinstance(obj, h5py.Dataset):
            print(f"{key_color}{name}{Color.reset}:")
            print(f"  shape={obj.shape} {range_str(obj, precision)} ({obj.dtype})")
            # print(type(obj.dtype)) # numpy.dtype
            if attrs:
                print(f"  attrs: {attrs}")
        else:
            if attrs:
                print(f"{name}:")
                print("  attrs:", list(attrs))

    root.visititems(PrintOnlyDataset)
    print("=" * len(s))


def h5py2rdict(d: h5py.Group):
    # h5py.Group <- h5py.File
    dst = {}
    _h5py2rdict(d, dst)
    return dst


def _h5py2rdict(d: h5py.Group, dst: D):
    for k, v in d.items():
        if type(v) == h5py.Group:
            dst[k] = {}
            _h5py2rdict(d[k], dst[k])
        elif type(v) == h5py.Dataset:
            dst[k] = v
        else:
            assert False


# https://docs.h5py.org/en/stable/high/file.html
# Accessing the File instance after the underlying file object has been closed will result in undefined behaviour.

# def rdict2h5py(d: h5py.Group):
#     # h5py.Group <- h5py.File
#     dst = h5py.Group()
#     _rdict2h5py(d, dst)
#     return dst


# def _rdict2h5py(d: h5py.Group, dst: dict):
#     for k, v in d.items():
#         if type(v) == h5py.Group:
#             dst[k] = {}
#             _rdict2h5py(d[k], dst[k])
#         elif type(v) == h5py.Dataset:
#             dst[k] = v
#         else:
#             assert False
