import os
import sys
from pathlib import Path

workspaceFolder = Path(__file__).absolute().parent.parent
os.chdir(workspaceFolder)
sys.path.append(str(workspaceFolder))
sys.path.append(str(workspaceFolder / "source"))

import argparse
from argparse import RawTextHelpFormatter
from pprint import pprint

import argset
from source.simulation import control

# fmt: off
parser = argparse.ArgumentParser(
    allow_abbrev=False,
    formatter_class=RawTextHelpFormatter,
    description=
"""Examples:
  $ python control.py --config-ctrl config/reacher2d_ctrl.json5
""",
)
parser.add_argument("--config-ctrl", type=str, required=True)
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--fix-xmap-size", type=float, metavar="S", help="xmap size")
parser.add_argument("--steps", type=int,  metavar="E", help="Time steps")
parser.add_argument("--save-anim", action="store_true")
parser.add_argument("--format", type=str, default="mp4", **argset.format_video)
args = parser.parse_args()
# fmt: on


control.main(**vars(args))
