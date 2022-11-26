import sys
from pathlib import Path

import numpy as np
import torch

import tool.util
from models.core import get_NewtonianVAE
from mypython.terminal import Color
from simulation.env import obs2img
from tool import argset, checker
from tool.params import Params, ParamsEval
from tool.util import Preferences


def load(path_model, cf_eval):
    d = tool.util.select_date(path_model)
    if d is None:
        sys.exit()
    weight_p = tool.util.select_weight(d)
    if weight_p is None:
        sys.exit()

    params = Params(Path(d, "params_bk.json5"))
    params_eval = ParamsEval(cf_eval)
    print("params (backup):")
    print(params)
    print()
    print("params eval:")
    print(params_eval)

    dtype: torch.dtype = getattr(torch, params_eval.dtype)

    checker.cuda(params.train.device)
    device = torch.device(params_eval.device if torch.cuda.is_available() else "cpu")

    model = get_NewtonianVAE(params.model, **params.raw_[params.model])

    model.cell.load_state_dict(torch.load(weight_p))
    model.train(params_eval.training)
    model.type(dtype)
    model.to(device)

    return model, d, weight_p, params, params_eval, dtype, device


def get_path_data(arg_path_data, d):
    if arg_path_data is not None:
        data_path = arg_path_data
    else:
        data_path = Preferences.get(d, "data_from")
        if data_path is None:
            Color.print("Specify the data path", c=Color.red)
            sys.exit()

    if Preferences.get(data_path, "id") != Preferences.get(d, "data_id"):
        Color.print(
            "warning: The dataset used for training may be different from the dataset you are about to use.",
            c=Color.coral,
        )

    return data_path
