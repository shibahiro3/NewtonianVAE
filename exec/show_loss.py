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
from source.view import show_loss

# fmt: off
parser = argparse.ArgumentParser(
    allow_abbrev=False,
    formatter_class=RawTextHelpFormatter,
    description=
"""Examples:
  $ python show_loss.py -c config/reacher2d.json5
  $ python show_loss.py -c config/point_mass.json5
""",
)
parser.add_argument("-c", "--config", type=str, required=True, **argset.config)
parser.add_argument("--start-iter", type=int, default=100, metavar="NUM", help="Number of initial iterations to draw Loss")
parser.add_argument("--format", type=str, default=["svg", "pdf", "png"], nargs="*", **argset.format_file)
args = parser.parse_args()
# fmt: on


show_loss.main(**vars(args))
