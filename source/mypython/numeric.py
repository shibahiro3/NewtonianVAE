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
