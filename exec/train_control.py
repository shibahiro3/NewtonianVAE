import os
import sys

os.chdir(os.pardir)  # workspaceFolder
sys.path.append("source")

import argparse
from argparse import RawTextHelpFormatter
from pprint import pprint

import argset

from control import train

# fmt: off
parser = argparse.ArgumentParser(allow_abbrev=False, formatter_class=RawTextHelpFormatter)
parser.add_argument("--config", type=str, default="config/reacher2d_ctrl.json5", **argset.config)
parser.add_argument("--path-model", type=str, default="environment/reacher2d/saves")
args = parser.parse_args()
# fmt: on


train.train(**vars(args))
