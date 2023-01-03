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
from source.simulation import control_pure

# fmt: off
parser = argparse.ArgumentParser(
    allow_abbrev=False,
    formatter_class=RawTextHelpFormatter,
    description=
"""Examples:
  $ python control_pure.py --config config/reacher2d.json5
""",
)
parser.add_argument("--config", type=str, required=True, **argset.config)
parser.add_argument("--goal-img", type=str, default="environments/reacher2d/goals/obs_green.npy", metavar="PATH", help="Goal image path (*.npy)")
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--fix-xmap-size", type=float, metavar="S", help="xmap size")
parser.add_argument("--steps", type=int, default=150, metavar="E", help="Time steps")
parser.add_argument("--alpha", type=float, default=0.4, metavar="α", help="P gain")
parser.add_argument("--save-anim", action="store_true")
parser.add_argument("--format", type=str, default="mp4", **argset.format_video)
args = parser.parse_args()
# fmt: on


control_pure.main(**vars(args))
