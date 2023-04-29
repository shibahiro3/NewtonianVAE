"""
python gmm.py
"""

import os
import sys

sys.path.append((os.pardir + os.sep) * 4)

from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.gridspec import GridSpec
from torch import Tensor, nn, optim

import mypython.ai.torchprob as tp
import mypython.plotutil as mpu
from mypython.ai.util import BatchIndices, to_np
from mypython.numeric import RemainingTime
from mypython.pyutil import s2dhms_str
from mypython.terminal import Color, Prompt


class MDN(tp.GMM):
    def __init__(self, D, K, dim_middle):
        super().__init__()

        self.K = K
        self.D = D
        self.dim_middle = dim_middle

        self.feature = nn.Sequential(
            nn.Linear(D, dim_middle),
            nn.Tanh(),
        )

        self.pi = nn.Sequential(
            nn.Linear(dim_middle, K),
            nn.Softmax(dim=-1),
        )
        self.mu = nn.Linear(dim_middle, D * K)
        self.sigma = nn.Linear(dim_middle, D * K)

    def forward(self, x: Tensor):
        x = self.feature(x)
        pi = self.pi(x)
        sigma = F.softplus(self.sigma(x))
        sigma = sigma.reshape(-1, self.K, self.D)
        mu = self.mu(x)
        mu = mu.reshape(-1, self.K, self.D)
        return pi, mu, sigma


class SimpleBatchData(BatchIndices):
    def __init__(self, data: Tensor, batch_size: int, device: torch.device):
        super().__init__(0, len(data), batch_size)
        self.data = data
        self.device = device

    def __next__(self):
        return self.data[super().__next__()].to(self.device)


def main():

    N, K, D = 2000, 3, 2
    batch_size = 200
    epochs = 2000
    device = torch.device("cpu")
    # device = torch.device("cuda")

    truth_pi = torch.tensor([0.3, 0.5, 0.2])
    truth_mu = torch.tensor([[-2, -1], [0.7, 0.3], [1.9, -0.4]])
    truth_sigma = torch.tensor([[0.4, 0.2], [0.2, 0.1], [0.3, 0.1]])

    pi = truth_pi.expand(N, K)
    mu = truth_mu.expand(N, K, D)
    sigma = truth_sigma.expand(N, K, D)

    print("pi shape:   ", pi.shape)
    print("mu shape:   ", mu.shape)
    print("sigma shape:", sigma.shape)

    X = tp.GMM(pi=pi, mu=mu, sigma=sigma).sample()

    print("Data scale")
    print(f"{X.min().item():+.4f}, {X.max().item():+.4f}")

    # ============================================================

    dataloader = SimpleBatchData(X, batch_size, device)

    prob = MDN(D=D, K=K, dim_middle=2 * D)
    prob.to(device)

    optimizer = optim.Adam(prob.parameters())

    record_Loss = []

    # tp.config.check_value = False  # A little bit faster
    remaining = RemainingTime(max=epochs * len(dataloader), size=50)
    for epoch in range(1, epochs + 1):
        for x in dataloader:

            # Objective function: -log p(x | Θ) = -SUM_k π_k log p(x | μ_k, Σ_k)  (Negative log-likelihood)
            # Since π_k, μ_k, Σ_k in π_k log p(x | μ_k, Σ_k) is inferred from x,
            # it is an implementation of p(x | x).
            L = -tp.log(prob, x).given(x).mean()

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

    sample = tp.GMM(pi=pi, mu=mu, sigma=sigma).sample().to(device)

    resample = prob.given(sample).sample()

    sample = to_np(sample)
    resample = to_np(resample)

    print("Sample shape:  ", sample.shape)
    print("Resample shape:", resample.shape)

    plt.title("Loss")
    plt.plot(record_Loss)

    plt.rcParams.update(
        {
            "lines.marker": "o",
            "lines.markersize": 6,
            "lines.markeredgecolor": "None",
            "lines.linestyle": "None",
        }
    )

    p = sns.jointplot(x=sample[:, 0], y=sample[:, 1], marginal_kws=dict(bins=100))
    p.fig.suptitle("Sample")
    plt.plot(truth_mu[..., 0], truth_mu[..., 1], color="red", label=r"truth $\mu$")
    plt.legend()

    p = sns.jointplot(x=resample[:, 0], y=resample[:, 1], marginal_kws=dict(bins=100))
    p.fig.suptitle("Resample")
    plt.plot(truth_mu[..., 0], truth_mu[..., 1], color="red", label=r"truth $\mu$")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
