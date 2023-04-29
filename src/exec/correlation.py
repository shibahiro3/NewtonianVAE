import common

common.set_path(__file__)

import argparse
from argparse import RawTextHelpFormatter

from newtonianvae import correlation

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
parser.add_argument("--episodes", type=int)
parser.add_argument("--all", action='store_true', help="Show correlations for all combinations")
parser.add_argument("--position-name", type=str, default="position")
parser.add_argument("--all-epochs", action='store_true')
parser.add_argument("--format", type=str, default=["svg", "pdf", "png"], nargs="*", **common.format_file)
args = parser.parse_args()
# fmt: on


correlation.correlation(**vars(args))
