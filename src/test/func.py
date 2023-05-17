if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.pardir + os.sep)


import math
import os
import pickle
import random
import re
import shutil
import sys
import threading
import time
import traceback
from datetime import datetime
from numbers import Real
from pathlib import Path
from pprint import pprint
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from torch import Tensor, nn, optim
from torchvision import models

from exec.train import cyclical_linear
from mypython.pyutil import function_test


# @function_test
def cyclical_linear_():
    y = []
    for x in range(1, 500):
        y.append(cyclical_linear(x, 40, 0, 1))

    # plt.ion()
    plt.plot(y)
    plt.show(block=False)
    plt.pause(0.01)

    time.sleep(3)


if __name__ == "__main__":
    cyclical_linear_()
