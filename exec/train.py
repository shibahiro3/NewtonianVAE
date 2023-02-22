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
from source.newtonianvae import train
from source.view import visualhandler

# fmt: off
parser = argparse.ArgumentParser(
    allow_abbrev=False,
    formatter_class=RawTextHelpFormatter,
    description=
"""if you use "--visual tensorboard":
  1. Another terminal: $ tensorboard --logdir="log_tb"
  2. Open the output URL (http://localhost:6006/) in a browser
  3. $ python train.py -c ... --visual tensorboard

if you use "--visual visdom":
  1. Another terminal: $ python -m visdom.server -port 8097
  2. Open the output URL (http://localhost:8097) in a browser
  3. $ python train.py -c ... --visual visdom

Examples:
  $ python train.py -c config/reacher2d.json5
  $ python train.py -c config/point_mass.json5
  $ python train.py -c config/reacher2d.json5 --visual visdom
""",
)
parser.add_argument("-c", "--config", type=str, required=True, **argset.config)
parser.add_argument("--resume", action="store_true", help="Load the model and resume learning")
parser.add_argument("--visual", type=str, choices=["tensorboard", "visdom"])
args = parser.parse_args()
# fmt: on


if args.visual is None:
    vh = visualhandler.VisualHandlerBase()
elif args.visual == "tensorboard":
    vh = visualhandler.TensorBoardVisualHandler(log_dir="log_tb")
elif args.visual == "visdom":
    vh = visualhandler.VisdomVisualHandler(port=8097)

argdict = vars(args)
argdict.pop("visual")
train.train(**argdict, vh=vh)
