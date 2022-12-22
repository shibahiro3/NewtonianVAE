import os
import sys

os.chdir(os.pardir)  # workspaceFolder
sys.path.append("source")

import argparse
from argparse import RawTextHelpFormatter
from pprint import pprint

import argset

from newtonianvae import reconstruct

# fmt: off
parser = argparse.ArgumentParser(allow_abbrev=False, formatter_class=RawTextHelpFormatter)
parser.add_argument("--config", type=str, default="config/reacher2d.json5", **argset.config)
parser.add_argument("--path-model", type=str, **argset.path_model)
parser.add_argument("--path-data", type=str, **argset.path_data)
parser.add_argument("--path-result", type=str, **argset.path_result)
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--fix-xmap-size", type=float, metavar="S", help="xmap size")
parser.add_argument("--save-anim", action="store_true")
parser.add_argument("--format", type=str, default="mp4")
args = parser.parse_args()
# fmt: on


reconstruct.reconstruction(**vars(args))
