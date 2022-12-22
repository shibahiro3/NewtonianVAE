import os
import sys

os.chdir(os.pardir)  # workspaceFolder
sys.path.append("source")

import argparse
from argparse import RawTextHelpFormatter
from pprint import pprint

import argset

from simulation import collect_data

# fmt: off
parser = argparse.ArgumentParser(allow_abbrev=False, formatter_class=RawTextHelpFormatter)
parser.add_argument("--config", type=str, default="config/reacher2d.json5", metavar="FILE", help="Configuration file\nYou need to write extarnal:\"data_path\".")
parser.add_argument("--episodes", type=int, default=1400)  # for train: 1000  + [e.g.] for validation: 200, for test: 200
parser.add_argument("--watch", type=str, choices=["render", "plt"], help="Check data without saving data. For rendering, you can choose to use OpenCV (render) or Matplotlib (plt).")
parser.add_argument("--save-anim", action="store_true")
parser.add_argument("--format", type=str, default="mp4")
args = parser.parse_args()
# fmt: on


collect_data.main(**vars(args))
