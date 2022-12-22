import os
import sys

os.chdir(os.pardir)  # workspaceFolder
sys.path.append("source")

import argparse
from argparse import RawTextHelpFormatter
from pprint import pprint

import argset

from simulation import control

# fmt: off
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--config", type=str, default="config/reacher2d.json5", **argset.config)
parser.add_argument("--path-model", type=str, **argset.path_model)
parser.add_argument("--path-result", type=str, **argset.path_result)
parser.add_argument("--goal-img", type=str, default="environment/reacher2d/observation_imgs/obs_green.npy", metavar="PATH", help="Goal image path (*.npy)")
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--fix-xmap-size", type=float, default=4, metavar="S", help="xmap size")
parser.add_argument("--steps", type=int, default=200, metavar="E", help="Time steps")
parser.add_argument("--save-anim", action="store_true")
parser.add_argument("--format", type=str, default="mp4")
parser.add_argument("--alpha", type=float, default=0.4, metavar="α", help="P gain")
args = parser.parse_args()
# fmt: on


control.main(**vars(args))
