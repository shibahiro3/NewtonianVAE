# only standard library

from __future__ import annotations

import builtins
import copy
import datetime
import errno
import fcntl
import inspect
import math
import os
import pickle
import pprint
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import typing
import unicodedata
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from posix import times_result
from pprint import pformat
from typing import Callable, Optional, Tuple

from mypython.terminal import Color


def is_number_type(x) -> bool:
    _tx = type(x)
    return _tx == int or _tx == float


def check_args_type(f: Callable, argname_value):
    """argname_value: ex. locals(), self.__dict__"""

    errs = []

    for arg_name, ttype in f.__annotations__.items():
        if arg_name in ("self", "return"):
            continue

        _value = argname_value[arg_name]
        _type = type(_value)
        # print(arg_name, ":", _value, _type.__name__, "|", ttype)
        if typing.get_origin(ttype) == typing.Union:
            _u_args = typing.get_args(ttype)
            if _type not in _u_args:
                elem = " or ".join([f'"{_ttype.__name__}"' for _ttype in _u_args])
                errs.append(f'The value {_value} specified for "{arg_name}" is not of type {elem}.')
        else:
            if _type != ttype:
                errs.append(
                    f'The value {_value} specified for "{arg_name}" is not of type "{ttype.__name__}".'
                )

    if len(errs) > 0:
        errs = ["  " + e for e in errs]
        raise TypeError("\n" + "\n".join(errs))

        # TODO: typing.Tuple


def s2dhms(sec):
    sec = math.floor(sec)
    d = math.floor(sec / 86400)
    _rest_sec = sec - d * 86400
    h = math.floor(_rest_sec / 3600)
    _rest_sec = _rest_sec - h * 3600
    m = math.floor(_rest_sec / 60)
    s = _rest_sec - m * 60
    return d, h, m, s


def s2dhms_str(sec, always_day=False):
    d, h, m, s = s2dhms(sec)
    if always_day:
        return f"{d} Days {h}:{m:0>2}:{s:0>2}"
    else:
        if abs(d) > 0:
            return f"{d} Days {h}:{m:0>2}:{s:0>2}"
        else:
            return f"{h}:{m:0>2}:{s:0>2}"


def add_version(path) -> Path:
    """
    for probabilistic result, etc.

    path: need suffix
    """

    path = Path(path)
    remove_suffix = str(path.with_suffix(""))
    suffix = path.suffix
    version = 1
    while True:
        s = Path(remove_suffix + f"_V{version}" + suffix)
        version += 1
        if not s.exists():
            break
    return s


class initialize:
    """
    Examples:
        @initialize.all_with(None)
        class Data:
            ...
    """

    @staticmethod
    def all_with(value, value_wrap=copy.copy):
        def _inner(cls):
            def init(self):
                for name in cls.__annotations__.keys():
                    if value_wrap is None:
                        setattr(self, name, value)
                    else:
                        setattr(self, name, value_wrap(value))

            setattr(cls, "__init__", init)
            return cls

        return _inner


def function_test(func):
    """
    @function_test
    def func...
    """

    def wrapper(*args, **kwargs):
        Color.print(f"\n=== [Start] {func.__name__} ===", c=Color.orange)
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_time = time.perf_counter() - start
        Color.print(f"=== [End]   {func.__name__}  Elapsed: {elapsed_time:.2f}s ===", c=Color.green)
        return result

    return wrapper


# def function_test(msg=""):
#     # 常にかっこ要る @function_test()

#     def _function_test(func):
#         def wrapper(*args, **kwargs):
#             Color.print(f"\n=== [Start] {func.__name__}", c=Color.orange)
#             start = time.perf_counter()
#             result = func(*args, **kwargs)
#             elapsed_time = time.perf_counter() - start
#             Color.print(f"=== [End]   {func.__name__}  Elapsed: {elapsed_time:.2f}s", c=Color.green)
#             return result

#         return wrapper

#     return _function_test


def run_once(func):
    """
    @run_once
    def func...
    """

    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return func(*args, **kwargs)

    wrapper.has_run = False
    return wrapper


def singleton_class(class_):
    """
    @singleton_class
    class Class...
    """

    instance = [
        None,
    ]

    def wrapper(*args, **kwargs):
        if instance[0] is None:
            instance[0] = class_(*args, **kwargs)
        return instance[0]

    return wrapper


def human_readable_byte(nbytes: int, bin=False) -> str:
    # torch.Tensor: b = x.element_size() * x.numel()

    assert type(nbytes) == int
    assert nbytes >= 0

    def _p(nbytes_, unit: str) -> str:
        if bin:
            return f"{nbytes_:.3f} {unit[0]}i{unit[1]}"
        else:
            return f"{nbytes_:.3f} {unit}"

    if bin:
        U = 1024
    else:
        U = 1000

    if nbytes < U:
        return f"{nbytes} B"

    nbytes /= U
    if nbytes < U:
        return _p(nbytes, "KB")

    nbytes /= U
    if nbytes < U:
        return _p(nbytes, "MB")

    nbytes /= U
    if nbytes < U:
        return _p(nbytes, "GB")
