import sys
from pathlib import Path

import numpy as np
import torch
import torch.utils
import torch.utils.data

import tool.plot_config
import tool.util
from models.core import get_NewtonianVAECell
from mypython.terminal import Prompt
from simulation.env import obs2img
from tool import argset, checker
from tool.dataloader import GetBatchData
from tool.params import Params, ParamsEval


def load_nvae(path_model, cf_eval):
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

    cell = get_NewtonianVAECell(params.model, **params.raw_[params.model])

    cell.load_state_dict(torch.load(weight_p))
    cell.train(params_eval.training)
    cell.type(dtype)
    cell.to(device)

    return cell, d, weight_p, params, params_eval, dtype, device
