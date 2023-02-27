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
  $ python correlation.py -c config/reacher2d.json5
"""
)
parser.add_argument("-c", "--config", type=str, required=True, **argset.config)
parser.add_argument("--episodes", type=int, default=50)
parser.add_argument("--all", action='store_true', help="Show correlations for all combinations")
parser.add_argument("--format", type=str, default=["svg", "pdf", "png"], nargs="*", **argset.format_file)
args = parser.parse_args()
# fmt: on


correlation.correlation(**vars(args))
