import common

common.set_path(__file__)

import argparse
from argparse import RawTextHelpFormatter

from newtonianvae import corr_epoch

# fmt: off
parser = argparse.ArgumentParser(
    allow_abbrev=False,
    formatter_class=RawTextHelpFormatter,
)
parser.add_argument("-c", "--config", type=str, required=True, **common.config)
parser.add_argument("--format", type=str, default=["svg", "pdf", "png"], nargs="*", **common.format_file)
args = parser.parse_args()
# fmt: on


corr_epoch.main(**vars(args))
