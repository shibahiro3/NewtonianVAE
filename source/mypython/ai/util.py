from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np


class BatchIdx:
    def __init__(self, start: int, stop: int, batch_size: int):
        assert stop - start >= batch_size

        self.N = stop - start
        self.BS = batch_size
        self.indexes = np.arange(start=start, stop=stop)
        self._reset()

    def __len__(self):
        return np.ceil(self.N / self.BS)

    def __iter__(self):
        return self

    def __next__(self):
        self.i += 1

        if self.i == self.__len__():
            self._reset()
            raise StopIteration()

        mask = self.indexes[self.i * self.BS : (self.i + 1) * self.BS]
        return mask

    def _reset(self):
        np.random.shuffle(self.indexes)
        self.i = -1
