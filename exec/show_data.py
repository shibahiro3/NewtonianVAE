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
from source.view import show_data

# fmt: off
parser = argparse.ArgumentParser(
    allow_abbrev=False,
    formatter_class=RawTextHelpFormatter,
    description=
"""You can check your own data sets""",
)
parser.add_argument("-c", "--config", type=str, required=True, **argset.config)
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--save-anim", action="store_true")
parser.add_argument("--format", type=str, default="mp4", **argset.format_video)
args = parser.parse_args()
# fmt: on


show_data.main(**vars(args))
