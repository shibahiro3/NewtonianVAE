import common

common.set_path(__file__)

import argparse
from argparse import RawTextHelpFormatter

from newtonianvae import reconstruct

# fmt: off
parser = argparse.ArgumentParser(
    allow_abbrev=False,
    formatter_class=RawTextHelpFormatter,
    description=
"""Examples:
  $ python reconstruct.py -c config/reacher2d.json5
""",
)
parser.add_argument("-c", "--config", type=str, required=True, **common.config)
parser.add_argument("--episodes", type=int, default=10) # If too large, get torch.cuda.OutOfMemoryError
parser.add_argument("--save-anim", action="store_true")
parser.add_argument("--shuffle", action="store_true")
parser.add_argument("--format", type=str, default="mp4", **common.format_video)
args = parser.parse_args()
# fmt: on


reconstruct.reconstruction(**vars(args))
