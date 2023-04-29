"""
References:
    https://notebook.community/hardmaru/pytorch_notebooks/mixture_density_networks

python gmm_cond.py
"""

import os
import sys

sys.path.append((os.pardir + os.sep) * 4)

from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.gridspec import GridSpec
from torch import Tensor, nn, optim
from typing_extensions import Self

import mypython.ai.torchprob as tp
import mypython.plotutil as mpu
from mypython.ai.util import BatchIndices, to_np
from mypython.numeric import RemainingTime
from mypython.pyutil import s2dhms_str
from mypython.terminal import Color, Prompt


class MDN(tp.GMM):
    def __init__(self, D: int, K: int, n_hidden: int):
        super().__init__()
        self.K = K
        self.D = D
        self.n_hidden = n_hidden

        self.z_h = nn.Linear(D, n_hidden)
        self.z_pi = nn.Linear(n_hidden, K)
        self.z_mu = nn.Linear(n_hidden, K * D)
        self.z_sigma = nn.Linear(n_hidden, K * D)

    def forward(self, x: Tensor):
        x = self.z_h(x)
        x = torch.tanh(x)
        pi = self.z_pi(x)
        pi = torch.softmax(pi, dim=-1)
        mu = self.z_mu(x)
        mu = mu.reshape(-1, self.K, self.D)
        sigma = self.z_sigma(x)
        sigma = sigma.reshape(-1, self.K, self.D)
        sigma = F.softplus(sigma)
        return pi, mu, sigma


def generate_data(n_samples: int) -> Tuple[Tensor, Tensor]:
    epsilon = torch.distributions.Normal(0, 1).sample((n_samples,))
    x_data = torch.distributions.Uniform(-10.5, 10.5).sample((n_samples,))
    y_data = 7 * torch.sin(0.75 * x_data) + 0.5 * x_data + epsilon
    return x_data, y_data


def generate_data2(n_samples: int) -> Tuple[Tensor, Tensor]:
    epsilon = torch.distributions.Normal(0, 1).sample((n_samples,))
    x_data = torch.distributions.Uniform(-8.0, 21.0).sample((n_samples,))
    y_data = 3 * torch.sin(0.5 * x_data) + 0.2 * x_data + epsilon
    return x_data, y_data


class SimpleBatchData(BatchIndices):
    def __init__(self, y: Tensor, x: Tensor, batch_size: int, device: torch.device):
        assert len(y) == len(x)
        super().__init__(0, len(y), batch_size)

        self.y = y
        self.x = x
        self.device = device

    def __next__(self):
        mask = super().__next__()
        return self.y[mask].to(self.device), self.x[mask].to(self.device)


def main():

    N, K, D = 2000, 5, 2
    batch_size = 200
    epochs = 2000
    device = torch.device("cpu")
    # device = torch.device("cuda")

    x_data1, y_data1 = generate_data(N)
    x_data2, y_data2 = generate_data2(N)

    X = torch.stack([x_data1, x_data2]).T
    Y = torch.stack([y_data1, y_data2]).T
    X, Y = Y, X

    print("Data scale")
    print("Y:", f"{Y.min().item():+.4f}, {Y.max().item():+.4f}")
    print("X:", f"{X.min().item():+.4f}, {X.max().item():+.4f}")

    whatever = False
    if whatever:
        # Check data
        fig, axes = plt.subplots(nrows=1, ncols=2, squeeze=False)
        ax = axes[0][0]
        ax.set_xlabel("X[0]")
        ax.set_ylabel("Y[0]")
        ax.scatter(X[:, 0], Y[:, 0], alpha=0.2, label="Original")
        ax = axes[0][1]
        ax.set_xlabel("X[1]")
        ax.set_ylabel("Y[1]")
        ax.scatter(X[:, 1], Y[:, 1], alpha=0.2, label="Original")
        mpu.legend_reduce(fig)
        plt.show()

    datascaler_y = tp.Scaler(Y.mean(dim=0), Y.std(dim=0))
    datascaler_x = tp.Scaler(X.mean(dim=0), X.std(dim=0))
    dataloader = SimpleBatchData(datascaler_y.pre(Y), datascaler_x.pre(X), batch_size, device)

    prob = MDN(D=D, K=K, n_hidden=20)
    prob.to(device)

    optimizer = torch.optim.Adam(prob.parameters())
    record_Loss = []

    # tp.config.check_value = False  # A little bit faster
    remaining = RemainingTime(max=epochs * len(dataloader), size=50)
    for epoch in range(1, epochs + 1):
        for y, x in dataloader:

            # -log p(y|x)
            L = -tp.log(prob, y).given(x).mean()

            optimizer.zero_grad()
            L.backward()
            optimizer.step()

            remaining.update()
            Prompt.print_one_line(
                (
                    f"Epoch {epoch} | "
                    f"Loss: {L:.4f} | "
                    f"Elapsed: {s2dhms_str(remaining.elapsed)} | "
                    f"Remaining: {s2dhms_str(remaining.time)} | "
                    f"ETA: {remaining.eta} "
                )
            )
            record_Loss.append(L.item())

    print()
    # ============================================================

    plt.title("Loss")
    plt.plot(record_Loss)
    plt.show()

    _X0_test = torch.linspace(X[:, 0].min(), X[:, 0].max(), N)
    _X1_test = torch.linspace(X[:, 1].min(), X[:, 1].max(), N)
    X_test = torch.stack([_X0_test, _X1_test]).T.to(device)

    Y_sampled = prob.given(datascaler_x.pre(X_test)).sample()
    Y_sampled = datascaler_y.post(Y_sampled)

    X_test = to_np(X_test)
    Y_sampled = to_np(Y_sampled)

    # ============================================================
    fig, axes = plt.subplots(nrows=1, ncols=2, squeeze=False)
    ax = axes[0][0]
    ax.set_xlabel("X[0]")
    ax.set_ylabel("Y[0]")
    ax.scatter(X[:, 0], Y[:, 0], alpha=0.2, label="Original")
    ax.scatter(X_test[:, 0], Y_sampled[:, 0], alpha=0.2, color="red", label="Sampled")
    ax = axes[0][1]
    ax.set_xlabel("X[1]")
    ax.set_ylabel("Y[1]")
    ax.scatter(X[:, 1], Y[:, 1], alpha=0.2, label="Original")
    ax.scatter(X_test[:, 1], Y_sampled[:, 1], alpha=0.2, color="red", label="Sampled")
    mpu.legend_reduce(fig)
    plt.show()
    # ============================================================


if __name__ == "__main__":
    main()
