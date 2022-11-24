import argparse
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

import tool.util
from models.core import NewtonianVAECell, NewtonianVAEDerivationCell
from mypython.pyutil import s2dhms_str
from tool import argset
from tool.dataloader import DataLoader
from tool.params import Params


def main():
    pass


if __name__ == "__main__":
    main()
