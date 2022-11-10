import argparse
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils
import torch.utils.data
from torch import nn, optim

import tool.util
from models.core import (
    CollectTimeSeriesData,
    NewtonianVAECell,
    NewtonianVAECellDerivation,
)
from mypython.pyutil import s2dhms_str
from tool import argset
from tool.dataloader import GetBatchData
from tool.params import Params


def main():
    pass


if __name__ == "__main__":
    main()
