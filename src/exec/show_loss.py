import common

common.set_path(__file__)

"""
References:
    https://www.tensorflow.org/tensorboard/dataframe_api?hl=ja
"""

import argparse
from argparse import RawTextHelpFormatter
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt

import tool.util
import view.plot_config
from mypython.ai import train
from tool import paramsmanager


def main():
    # fmt: off
    description = \
"""\
Show loss figure
  Save: Press S key on a window

Examples:
  $ python show_loss.py -c config/reacher2d.json5
  $ python show_loss.py -c config/point_mass.json5 --mode epoch --start-iter 5
"""
    parser = argparse.ArgumentParser(allow_abbrev=False, formatter_class=RawTextHelpFormatter, description=description)
    parser.add_argument("-c", "--config", type=str, required=True, **common.config)
    parser.add_argument("--mode", type=str, default="epoch", choices=["batch", "epoch"])
    parser.add_argument("--start-iter", type=int, default=1, metavar="NUM", help="Number of initial iterations or ephocs to draw Loss")
    parser.add_argument("--format", type=str, default=common.default_fig_formats, nargs="*", **common.format_file)
    parser.add_argument("--no-window", action='store_true', help="Save instantly without displaying a window")
    args = parser.parse_args()
    # fmt: on

    show_loss(**vars(args))


def show_loss(
    config: str,
    start_iter: int,
    format: List[str],
    mode: str,
    no_window: bool = False,
):

    assert start_iter > 0

    view.plot_config.apply()
    plt.rcParams.update(
        {
            "figure.figsize": (11.39, 3.9),
            "figure.subplot.left": 0.05,
            "figure.subplot.right": 0.98,
            "figure.subplot.bottom": 0.15,
            "figure.subplot.top": 0.85,
            "figure.subplot.wspace": 0.4,
        }
    )

    params_path = paramsmanager.Params(config).path
    manage_dir = tool.util.select_date(params_path.saves_dir, no_weight_ok=False, min_epoch=5)
    if manage_dir is None:
        return

    train.show_loss(
        manage_dir=manage_dir,
        results_dir=params_path.results_dir,
        format=format,
        mode=mode,
        start_iter=start_iter,
        no_window=no_window,
    )


if __name__ == "__main__":
    main()
