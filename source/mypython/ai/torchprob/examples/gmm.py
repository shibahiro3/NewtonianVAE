"""
python gmm.py
"""

import os
import sys

sys.path.append((os.pardir + os.sep) * 4)


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from torch import Tensor, nn, optim

import mypython.ai.torchprob as tp
from mypython.ai.util import BatchIdx
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


class SimpleBatchData(BatchIdx):
    def __init__(self, data: Tensor, batch_size: int):
        super().__init__(0, len(data), batch_size)
        self.data = data

    def __next__(self):
        return self.data[super().__next__()]


def main():
    N, K, D = 5000, 3, 2

    truth_pi = torch.tensor([0.3, 0.5, 0.2])
    truth_mu = torch.tensor([[-2, -1], [0.7, 0.3], [1.9, -0.4]])
    truth_sigma = torch.tensor([[0.4, 0.2], [0.2, 0.1], [0.3, 0.2]])

    pi = truth_pi.repeat(N, 1)
    mu = truth_mu.repeat(N, 1, 1)
    sigma = truth_sigma.repeat(N, 1, 1)

    print("pi shape:   ", pi.shape)
    print("mu shape:   ", mu.shape)
    print("sigma shape:", sigma.shape)

    X = tp.GMM(pi=pi, mu=mu, sigma=sigma).sample()
    dataloader = SimpleBatchData(X, 100)

    prob = MDN(D=D, K=K, dim_middle=2 * D)
    optimizer = optim.Adam(prob.parameters())

    record_Loss = []

    for epoch in range(1, 1000 + 1):
        for x in dataloader:

            # Objective function: -log p(x | Θ) = -SUM_k π_k log p(x | μ_k, Σ_k)  (Negative log-likelihood)
            # Since π_k, μ_k, Σ_k in π_k log p(x | μ_k, Σ_k) is inferred from x,
            # it is an implementation of p(x | x).
            L = -tp.log(prob, x, x).mean()

            optimizer.zero_grad()
            L.backward()
            optimizer.step()

            Prompt.print_one_line(f"Epoch {epoch} | Loss: {L:.4f} ")
            record_Loss.append(L.item())

    print()

    # ============================================================

    sample = tp.GMM(pi=pi, mu=mu, sigma=sigma).sample()
    resample = prob.cond(sample).sample()

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
