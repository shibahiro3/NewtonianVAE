import os
import sys

os.chdir(os.pardir)  # workspaceFolder
sys.path.append("source")

import argparse
from argparse import RawTextHelpFormatter
from pprint import pprint

import argset

from view import show_data

# fmt: off
parser = argparse.ArgumentParser(allow_abbrev=False,formatter_class=RawTextHelpFormatter, description="You can check your own data sets")
parser.add_argument("--path-data", type=str, **argset.path_data, required=True) # environment/reacher2d/data_operate
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--save-anim", action="store_true")
parser.add_argument("--output", type=str, help="Path of the video to be saved (Extension is .mp4, etc.)")
args = parser.parse_args()
# fmt: on


show_data.main(**vars(args))
