import common

common.set_path(__file__)

import argparse
from argparse import RawTextHelpFormatter

from view import show_data

# fmt: off
parser = argparse.ArgumentParser(
    allow_abbrev=False,
    formatter_class=RawTextHelpFormatter,
    description=
"""Check datasets from file""",
)
parser.add_argument("-c", "--config", type=str, required=True, **common.config)
parser.add_argument("--episodes", type=int, required=True, help="If nor specified, it will load all as batch")
parser.add_argument("--shuffle", action="store_true")
parser.add_argument("--data-type", type=str, default="train", choices=("train", "valid", "test"))
parser.add_argument("--env", type=str)
parser.add_argument("--position-name", type=str, default="position")
parser.add_argument("--position-title", type=str, default="Position")
parser.add_argument("--save-path", type=str)
args = parser.parse_args()
# fmt: on


show_data.main(**vars(args))
