import os
import sys

os.chdir(os.pardir)  # workspaceFolder
sys.path.append("source")

import argparse
from argparse import RawTextHelpFormatter
from pprint import pprint

import argset

from newtonianvae import train
from view import train_visualhandler

# fmt: off
parser = argparse.ArgumentParser(allow_abbrev=False, formatter_class=RawTextHelpFormatter)
parser.add_argument("--config", type=str, default="config/reacher2d.json5", **argset.config)
parser.add_argument("--resume", action="store_true", help="Load the model and resume learning")
parser.add_argument("--visual", type=str, choices=["tensorboard", "visdom"])
args = parser.parse_args()
# fmt: on


if args.visual is None:
    vh = train_visualhandler.VisualHandlerBase()
elif args.visual == "tensorboard":
    # 1. Another console: $ tensorboard --logdir="log_tb"
    # 2. Open the output URL (http://localhost:6006/) in a browser
    # 3. $ python train.py --visual tensorboard
    vh = train_visualhandler.TensorBoardVisualHandler(log_dir="log_tb")
elif args.visual == "visdom":
    # 1. Another console: $ python -m visdom.server -port 8097
    # 2. Open the output URL (http://localhost:8097) in a browser
    # 3. $ python train.py --visual visdom
    vh = train_visualhandler.VisdomVisualHandler(port=8097)


argdict = vars(args)
argdict.pop("visual")
train.train(**argdict, vh=vh)

"""
Examples:
python train.py
python train.py --config "config/point_mass.json5"
python train.py --visual visdom
"""
