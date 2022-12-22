import os
import sys

os.chdir(os.pardir)  # workspaceFolder
sys.path.append("source")

import argparse
from argparse import RawTextHelpFormatter
from pprint import pprint

import argset

from newtonianvae import show_loss

# fmt: off
parser = argparse.ArgumentParser(allow_abbrev=False, formatter_class=RawTextHelpFormatter)
parser.add_argument("--path-model", type=str, default="environment/reacher2d/saves", **argset.path_model)
parser.add_argument("--path-result", type=str, default="environment/reacher2d/results", **argset.path_result)
parser.add_argument("--start-iter", type=int, default=100, metavar="NUM")
parser.add_argument("--format", type=str, default=["svg", "pdf"], nargs="*", **argset.fotmat_file)
args = parser.parse_args()
# fmt: on


show_loss.main(**vars(args))
