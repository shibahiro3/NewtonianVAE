import common

common.set_path(__file__)

import argparse
from argparse import RawTextHelpFormatter

from simulation import control_pure

# fmt: off
parser = argparse.ArgumentParser(
    allow_abbrev=False,
    formatter_class=RawTextHelpFormatter,
    description=
"""Examples:
  $ python correlation.py -c config/reacher2d.json5
"""
)
parser.add_argument("-c", "--config", type=str, required=True, **common.config)
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--save-anim", action="store_true")
parser.add_argument("--alpha", type=float, metavar="P_GAIN", default=1.0)
parser.add_argument("--steps", type=int, metavar="TIME_STEPS", default=150)
parser.add_argument("--format", type=str, default="mp4", **common.format_video)
args = parser.parse_args()
# fmt: on


control_pure.main(**vars(args))
