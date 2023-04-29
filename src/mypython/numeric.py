import datetime
import time
from datetime import datetime, timedelta
from typing import Callable, Optional, Tuple

import numpy as np


class MovingAverage:
    def __init__(self, size):
        assert size > 0

        self.reset(size)

    def __call__(self, x_n):
        self._x_buf[self._add_position] = x_n
        self._add_position = (self._add_position + 1) % self._size

        if self._first_cnt < self._size:
            self._first_cnt += 1

        self._x_mean = self._x_buf.sum() / self._first_cnt

        return self._x_mean

    def get(self):
        return self._x_mean

    def reset(self, size):
        self._size = size
        self._x_buf = np.zeros((self._size,))
        self._add_position = 0
        self._x_mean = 0
        self._first_cnt = 0


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


class MovingAverageTime(MovingAverage):
    def __init__(self, size):
        super().__init__(size)
        self.time_prev = time.perf_counter()

    def update(self):
        now = time.perf_counter()
        ret = super().__call__(now - self.time_prev)
        self.time_prev = now
        return ret


def divisors(n):
    """
    Return:
        sorted divisors
    """
    assert type(n) == int and n > 0

    lower_divisors, upper_divisors = [], []
    i = 1
    while i * i <= n:
        if n % i == 0:
            lower_divisors.append(i)
            if i != n // i:
                upper_divisors.append(n // i)
        i += 1
    return lower_divisors + upper_divisors[::-1]
