import common

common.set_path(__file__)

import argparse
from argparse import RawTextHelpFormatter

from view import show_loss

# fmt: off
parser = argparse.ArgumentParser(
    allow_abbrev=False,
    formatter_class=RawTextHelpFormatter,
    description=
"""Examples:
  $ python show_loss.py -c config/reacher2d.json5
  $ python show_loss.py -c config/point_mass.json5 --mode epoch --start-iter 5
""",
)
parser.add_argument("-c", "--config", type=str, required=True, **common.config)
parser.add_argument("--mode", type=str, default="epoch", choices=["batch", "epoch"])
parser.add_argument("--start-iter", type=int, default=1, metavar="NUM", help="Number of initial iterations or ephocs to draw Loss")
parser.add_argument("--format", type=str, default=["svg", "pdf", "png"], nargs="*", **common.format_file)
args = parser.parse_args()
# fmt: on


show_loss.main(**vars(args))
