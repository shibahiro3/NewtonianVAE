"""
References:
    https://www.tensorflow.org/tensorboard/dataframe_api?hl=ja
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

import mypython.plotutil as mpu
import tool.util
import view.plot_config
from mypython.ai.train import show_loss
from mypython.valuewriter import ValueWriter
from tool import paramsmanager

view.plot_config.apply()
try:
    import view._plot_config

    view._plot_config.apply()
except:
    pass


def main(
    config: str,
    start_iter: int,
    format: List[str],
    mode: str,
):
    assert start_iter > 0

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
    manage_dir = tool.util.select_date(params_path.saves_dir, no_weight_ok=False)
    if manage_dir is None:
        return

    show_loss(
        manage_dir=manage_dir,
        results_dir=params_path.results_dir,
        format=format,
        mode=mode,
        start_iter=start_iter,
    )
