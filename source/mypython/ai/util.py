import copy
import datetime
import json
import os
import random
import sys
import time
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class BatchIdx:
    def __init__(self, startN: int, stopN: int, BS: int):
        assert BS < stopN - startN

        self.N = stopN - startN
        self.BS = BS
        self.indexes = np.arange(start=startN, stop=stopN)
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
