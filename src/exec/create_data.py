import common

common.set_path(__file__)

import argparse
from argparse import RawTextHelpFormatter

from simulation import create_data

# fmt: off
parser = argparse.ArgumentParser(
    allow_abbrev=False,
    formatter_class=RawTextHelpFormatter,
    description=
"""Examples:
  $ python create_data.py -c config/reacher2d.json5 --mode show-plt
  $ python create_data.py -c config/reacher2d.json5 --mode save-data --save-dir data/reacher2d 
""",
)
parser.add_argument("-c", "--config", type=str, required=True, metavar="FILE", help="Configuration file\nYou need to write extarnal:\"data_path\".")
parser.add_argument("--mode", type=str, required=True, choices=("show-plt", "show-render", "save-data", "save-anim"))
parser.add_argument("--episodes", type=int, default=1400)  # for train: 1000  + [e.g.] for validation: 200, for test: 200
parser.add_argument("--save-dir", type=str, help="For data and animation")
parser.add_argument("--format", type=str, default="mp4", **common.format_video)
args = parser.parse_args()
# fmt: on


create_data.main(**vars(args))
