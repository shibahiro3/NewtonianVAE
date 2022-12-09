import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot

import mypython.ai.torchprob.debug as tp_debug
from models.core import get_NewtonianVAECell
from tool import argset
from tool.paramsmanager import Params

parser = argparse.ArgumentParser(allow_abbrev=False)
argset.cf(parser)
_args = parser.parse_args()


class Args:
    cf = _args.cf


args = Args()


def main():
    params = Params(args.cf)
    cell = get_NewtonianVAECell(params.model, **params.raw_[params.model])
    # cell.train()
    cell.force_training = True

    BS = params.train.batch_size
    # BS = 1
    print(type(cell).__name__)

    I_t = torch.randn(BS, 3, 64, 64)
    x_tn1 = torch.randn(BS, cell.dim_x)
    u_tn1 = torch.randn(BS, cell.dim_x)
    v_tn1 = torch.randn(BS, cell.dim_x)
    dt = torch.full((BS, 1), 0.1)

    # writer = SummaryWriter("log")

    # class Model(nn.Module):
    #     def __init__(self):
    #         super().__init__()
    #         self.cell = cell

    #     def forward(self, I_t, x_tn1, u_tn1, v_tn1, dt):
    #         E, E_ll, E_kl, x_t, v_t = self.cell(I_t, x_tn1, u_tn1, v_tn1, dt)
    #         return E

    # model = Model()

    # writer.add_graph(cell, (I_t, x_tn1, u_tn1, v_tn1, dt), use_strict_trace=False)
    # writer.close()

    E, E_ll, E_kl, x_t, v_t = cell(I_t, x_tn1, u_tn1, v_tn1, dt)

    make_dot(E, params=dict(cell.named_parameters())).render(
        Path("log_model", type(cell).__name__ + ".gv"), format="svg"
    )

    tp_debug.check_dist_model(cell)


if __name__ == "__main__":
    main()
