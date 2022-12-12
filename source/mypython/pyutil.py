import argparse
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
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from posix import times_result
from pprint import pformat
from typing import Optional

from mypython.numeric import MovingAverage


def is_number_type(x):
    _tx = type(x)
    return _tx == int or _tx == float


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
