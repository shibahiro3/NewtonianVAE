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
from source.simulation import collect_data

# fmt: off
parser = argparse.ArgumentParser(
    allow_abbrev=False,
    formatter_class=RawTextHelpFormatter,
    description=
"""Examples:
  $ python create_data.py --config config/reacher2d.json5
  $ python create_data.py --config config/reacher2d.json5 --watch render
  $ python create_data.py --config config/reacher2d.json5 --watch plt
  $ python create_data.py --config config/reacher2d.json5 --watch plt --save-anim
  $ python create_data.py --config config/point_mass.json5
""",
)
parser.add_argument("--config", type=str, required=True, metavar="FILE", help="Configuration file\nYou need to write extarnal:\"data_path\".")
parser.add_argument("--episodes", type=int, default=1400)  # for train: 1000  + [e.g.] for validation: 200, for test: 200
parser.add_argument("--watch", type=str, choices=["render", "plt"], help="Check data without saving data. For rendering, you can choose to use OpenCV (render) or Matplotlib (plt).")
parser.add_argument("--save-anim", action="store_true")
parser.add_argument("--format", type=str, default="mp4", **argset.format_video)
args = parser.parse_args()
# fmt: on


collect_data.main(**vars(args))
