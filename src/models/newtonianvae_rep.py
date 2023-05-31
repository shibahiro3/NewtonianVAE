"""
Reproduction of NewtonianVAE paper about encoder decoder

B.1. Architectures
All models used the encoder and decoder from Ha
and Schmidhuber [21] , except for the point mass envi-
ronment, where we use a spatial broadcast decoder [48].

[21]: D. Ha and J. Schmidhuber. World models. In NeurIPS, 2018.

[48]: N. Watters, L. Matthey, C. P. Burgess, and A. Lerchner.
Spatial Broadcast Decoder: A Simple Architecture for
Learning Disentangled Representations in VAEs. In
ICLR Workshop, 2019.

"""

from numbers import Real
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import NumberType, Tensor, nn

from . import limiter


class VanillaEncoder(nn.Module):
    r"""
    Input:  (N, 3, 64, 64)
    Output: (N, dim_output)

    References:
        https://github.com/ctallec/world-models/blob/d6abd9ce97409734a766eb67ccf0d1967ba9bf0c/models/vae.py#L32
        https://github.com/Kaixhin/PlaNet/blob/28c8491bc01e8f1b911300749e04c308c03db051/models.py#L150
    """

    def __init__(
        self,
        dim_output: int,
        img_channels: int = 3,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(img_channels, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        self.last = nn.Identity() if dim_output == 1024 else nn.Linear(1024, dim_output)
        # 1024 = 2 * 2 * 256

    def forward(self, x: Tensor):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        x = x.reshape(x.size(0), -1)
        x = self.last(x)
        return x


class VanillaDecoder(nn.Module):
    r"""
    Input:  (N, dim_input)
    Output: (N, img_channels, 64, 64)

    References:
        https://github.com/ctallec/world-models/blob/d6abd9ce97409734a766eb67ccf0d1967ba9bf0c/models/vae.py#L10
    """

    def __init__(
        self,
        dim_input: int,
        img_channels: int = 3,
        last_fn: str = "",
    ):
        super().__init__()

        self.fc = nn.Linear(dim_input, 1024)
        self.deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, img_channels, 6, stride=2)
        self.last_fn = getattr(limiter, last_fn, lambda x: x)

    def forward(self, x: Tensor):
        x = F.relu(self.fc(x))
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = self.last_fn(self.deconv4(x))
        return x


class SpatialBroadcastDecoder(nn.Module):
    r"""
    Input:  (N, dim_input)
    Output: (N, img_channels, 64, 64)

    References:
        Spatial Broadcast Decoder https://arxiv.org/abs/1901.07017

    References for implementation:
        https://github.com/dfdazac/vaesbd/blob/master/model.py#L6
    """

    def __init__(
        self,
        *,
        dim_input: int,
        img_channels: int = 3,
        last_fn: str = "",
    ) -> None:
        super().__init__()

        a = np.linspace(-1, 1, 64)
        b = np.linspace(-1, 1, 64)
        x_grid, y_grid = np.meshgrid(a, b)
        x_grid = torch.from_numpy(x_grid)
        y_grid = torch.from_numpy(y_grid)
        # Add as constant, with extra dims for N and C
        self.x_grid: Tensor
        self.y_grid: Tensor
        self.register_buffer("x_grid", x_grid.reshape((1, 1) + x_grid.shape))
        self.register_buffer("y_grid", y_grid.reshape((1, 1) + y_grid.shape))

        self.dec_convs = nn.Sequential(
            nn.Conv2d(dim_input + 2, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, img_channels, 3, padding=1),
        )

        self.last_fn = getattr(limiter, last_fn, lambda x: x)

    def forward(self, x: Tensor):
        N = x.size(0)
        # View z as 4D tensor to be tiled across new H and W dimensions
        x = x.reshape(x.shape + (1, 1))  # (N, D, 1, 1)

        # Tile across to match image size
        x = x.expand(-1, -1, 64, 64)  # (N, D, 64, 64)

        # Expand grids to batches and concatenate on the channel dimension
        x = torch.cat(
            [
                self.x_grid.expand(N, -1, -1, -1),
                self.y_grid.expand(N, -1, -1, -1),
                x,
            ],
            dim=1,
        )  # (N, D+2, 64, 64)

        x = self.last_fn(self.dec_convs(x))
        return x
