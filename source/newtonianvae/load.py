import sys
from pathlib import Path

import numpy as np
import torch

import tool.util
from models.core import get_NewtonianVAE
from simulation.env import obs2img
from tool import argset, checker
from tool.params import Params, ParamsEval


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
