import argparse
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn

import tool.util
from models.core import get_NewtonianVAE
from mypython.terminal import Color
from simulation.env import obs2img
from tool import checker
from tool.paramsmanager import Params, ParamsEval
from tool.util import Preferences


def creator(
    root: str,
    model_place,
    model_name: str,
    model_params: dict,
    resume: bool = False,
):
    datetime_now = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    managed_dir = Path(root, datetime_now)
    weight_dir = Path(managed_dir, "weight")
    weight_dir.mkdir(parents=True, exist_ok=True)

    ModelType = getattr(model_place, model_name)
    model: nn.Module = ModelType(**model_params)

    if resume:
        print('You chose "resume". Select a model to load.')
        resume_manage_dir = tool.util.select_date(root)
        resume_weight_path = tool.util.select_weight(resume_manage_dir)
        model.load_state_dict(torch.load(resume_weight_path))
    else:
        resume_weight_path = None

    return model, managed_dir, weight_dir, resume_weight_path


def load(path_model, cf_eval):
    d = tool.util.select_date(path_model)
    if d is None:
        sys.exit()
    weight_path = tool.util.select_weight(d)
    if weight_path is None:
        sys.exit()

    params_path = Path(d, "params_saved.json5")
    params = Params(params_path)
    params_eval = ParamsEval(cf_eval)
    Color.print("params path     :", params_path)
    Color.print("params eval path:", cf_eval)

    dtype: torch.dtype = getattr(torch, params_eval.dtype)

    checker.cuda(params.train.device)
    device = torch.device(params_eval.device if torch.cuda.is_available() else "cpu")

    model = get_NewtonianVAE(params.model, **params.raw_[params.model])

    model.cell.load_state_dict(torch.load(weight_path))
    model.train(params_eval.training)
    model.type(dtype)
    model.to(device)

    return model, d, weight_path, params, params_eval, dtype, device


def get_path_data(arg_path_data, params: Params):
    if arg_path_data is not None:
        data_path = arg_path_data
    else:
        data_path = params.external.data_path
        if data_path is None:
            Color.print("Specify the data path", c=Color.red)
            sys.exit()

    if Preferences.get(data_path, "id") != params.external.data_id:
        Color.print(
            "warning: The dataset used for training may be different from the dataset you are about to use.",
            c=Color.coral,
        )

    return data_path
