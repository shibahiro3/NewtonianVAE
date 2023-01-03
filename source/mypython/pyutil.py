from __future__ import annotations

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
from pathlib import Path
from posix import times_result
from pprint import pformat
from typing import Callable, Optional, Tuple

import numpy as np
from typing_extensions import Self

from mypython.numeric import MovingAverage


def is_number_type(x):
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
    if d > 0:
        return f"{d} Days {h}:{m:0>2}:{s:0>2}"
    else:
        if always_day:
            return f"{d} Days {h}:{m:0>2}:{s:0>2}"
        else:
            return f"{h}:{m:0>2}:{s:0>2}"


def add_version(path) -> Path:
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


class Seq:
    def __init__(self, start=0, step=1, start_now=False) -> None:
        self.step = step

        if start_now:
            self._i = start
        else:
            self._i = start - step

    def next(self):
        self._i += self.step
        # self._i_ret = self._i
        return self._i

    @property
    def now(self):
        return self._i


class Seq2:
    def __init__(self, size: int, ratio_index: Tuple[int, int], start=0, lazy=False) -> None:
        """| a, b | a, b | ..."""

        _gcd = math.gcd(*ratio_index)
        _a = ratio_index[0] // _gcd
        _b = ratio_index[1] // _gcd

        self._start = start
        self._step_a = _a
        self._step_b = _b
        self._size = size
        self._length = (_a + _b) * (size - 1) + _a
        self._lazy = lazy

        if not self._lazy:
            self._i = start - _b
        else:
            self._i = start

    @property
    def a(self):
        if not self._lazy:
            self._i += self._step_b
            return self._i
        else:
            return self._i + self._step_b

    @property
    def b(self):
        if not self._lazy:
            self._i += self._step_a
            return self._i
        else:
            return self._i + self._step_a

    @property
    def length(self):
        return self._length

    @property
    def size(self):
        return self._size

    def update(self):
        self._i += self._step_a + self._step_b

    @staticmethod
    def share_length(*seq2s: Seq2):
        for e in seq2s:
            assert e._lazy == False

        lengths = np.array([e.length for e in seq2s])
        length = np.lcm.reduce(lengths)

        for i, e in enumerate(seq2s):
            coef = length // lengths[i]
            e._step_a *= coef
            e._step_b *= coef
            e._length = (e._step_a + e._step_b) * (e._size - 1) + e._step_a
            e._i = e._start - e._step_b

        return length


class RemainingTime:
    def __init__(self, max: Optional[int] = None, size=10) -> None:
        """
        Args:
            max: total number of iterations (number of times update is called)
        """

        self._max = max
        self._cnt = 0
        self._time = 0
        self._ave = MovingAverage(size=size)
        self._time_prev = time.perf_counter()
        self._time_start = self._time_prev

    def update(self, max: Optional[int] = None):
        """
        max: 0 is ok but only for elapsed
        """

        assert type(max) == int or type(self._max) == int
        if type(max) != int:
            max = self._max

        self._cnt += 1

        self._time_now = time.perf_counter()
        # delta = self._time_now - self._time_prev
        delta = self._ave(self._time_now - self._time_prev)

        self._time_prev = self._time_now
        self._time = delta * (max - self._cnt)

        return self

    @property
    def time(self):
        """Returns: [seconds]"""
        return self._time

    @property
    def eta(self):
        return (datetime.now() + timedelta(seconds=self.time)).strftime("%m/%d %H:%M")

    @property
    def elapsed(self):
        """Returns: [seconds]"""
        return self._time_now - self._time_start


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
