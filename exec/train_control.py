import os
import sys
from pathlib import Path

workspaceFolder = Path(__file__).absolute().parent.parent
os.chdir(workspaceFolder)
sys.path.append(str(workspaceFolder))
sys.path.append(str(workspaceFolder / "source"))


import argparse
from argparse import RawTextHelpFormatter

import argset
from source.controller import train

# fmt: off
parser = argparse.ArgumentParser(
    allow_abbrev=False,
    formatter_class=RawTextHelpFormatter,
    description=
"""Examples:
  $ python train_control.py --config config/reacher2d.json5 --config-ctrl config/reacher2d_ctrl.json5
""",
)
parser.add_argument("--config", type=str, required=True, **argset.config)
parser.add_argument("--config-ctrl", type=str, required=True)
args = parser.parse_args()
# fmt: on


train.train(**vars(args))
