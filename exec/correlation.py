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
from source.newtonianvae import correlation

# fmt: off
parser = argparse.ArgumentParser(
    allow_abbrev=False,
    formatter_class=RawTextHelpFormatter,
    description=
"""Examples:
  $ python correlation.py -c config/reacher2d.json5 --env-domain reacher2d
"""
)
parser.add_argument("-c", "--config", type=str, required=True, **argset.config)
parser.add_argument("--episodes", type=int, default=50)
parser.add_argument("--env-domain", type=str, metavar="ENV", help="Example: reacher2d, point_mass")
parser.add_argument("--fix-xmap-size", type=float, metavar="S", help="xmap size")
parser.add_argument("--format", type=str, default=["svg", "pdf", "png"], nargs="*", **argset.format_file)
args = parser.parse_args()
# fmt: on


correlation.correlation(**vars(args))
