r"""Encoders & Decoders (without probabilistic model)

Constructor:
    Encoder
        arg1: dim_output (Dimensions of latent space)
        arg2, ... : default args
    Decoder
        arg1: dim_input (Dimensions of latent space)
        arg2, ... : default args


References:
    https://github.com/ctallec/world-models/blob/master/models/vae.py
    https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
    https://github.com/dfdazac/vaesbd/blob/master/model.py
"""


from numbers import Real
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import NumberType, Tensor, nn
from torchvision import models

from models import from_stable_diffusion
from mypython.ai import nnio
from mypython.terminal import Color


class VisualEncoder64(nn.Module):
    r"""
    (N, 3, 64, 64) -> (N, dim_output)

    References for implementation:
        https://github.com/ctallec/world-models/blob/master/models/vae.py#L32
    """

    def __init__(
        self,
        *,
        dim_output: int,
        activation: str = "ReLU",
        channels: list = [3, 32, 64, 128, 256],
        kernels: list = [4, 4, 4, 4],
        strides: list = [2, 2, 2, 2],
        img_size: list = [64, 64],
        debug: bool = False,
    ) -> None:
        super().__init__()

        assert len(kernels) == len(channels) - 1
        assert len(kernels) == len(strides)

        self.debug = debug

        Activation = getattr(nn, activation)

        self.seq = nn.Sequential()
        for i in range(len(kernels)):
            self.seq.append(
                nn.Conv2d(channels[i], channels[i + 1], kernel_size=kernels[i], stride=strides[i])
            )
            self.seq.append(Activation())

        s = (3, img_size[0], img_size[1])
        OC, OH, OW = nnio.last_size(self.seq, s)
        last_dim = OC * OH * OW

        if self.debug:
            print(f"# {self.__class__.__name__} debug #")
            print("  ", "OC, OH, OW:", OC, OH, OW)
            print("  ", "last dim:", last_dim)

        self.last = nn.Identity() if dim_output == last_dim else nn.Linear(last_dim, dim_output)

    def forward(self, x: Tensor):
        """"""

        if self.debug:
            print(f"=== {self.__class__.__name__} ===")
            print(x.shape)
            for e in self.seq:
                x = e(x)
                print(x.shape)
        else:
            x = self.seq(x)

        # print(x.shape)
        x = x.flatten(start_dim=1)
        # print(x.shape)
        x = self.last(x)
        return x


class VisualDecoder64(nn.Module):
    r"""
    (N, dim_input) -> (N, 3, 64, 64)

    References for implementation:
        https://github.com/ctallec/world-models/blob/master/models/vae.py#L10
    """

    def __init__(
        self,
        *,
        dim_input: int,
        dim_middle: int,
        activation: str = "ReLU",
        channels: list = [3, 32, 64, 128],
        kernels: list = [6, 6, 5, 5],
        strides: list = [2, 2, 2, 2],
        debug: bool = False,
    ):
        super().__init__()

        assert len(kernels) == len(channels)
        assert len(kernels) == len(strides)

        self.debug = debug

        self.dim_middle = dim_middle

        channels.append(dim_middle)
        channels = list(reversed(channels))
        kernels = list(reversed(kernels))
        strides = list(reversed(strides))

        self.first = nn.Linear(dim_input, dim_middle)

        Activation = getattr(nn, activation)

        self.seq = nn.Sequential()
        for i in range(len(kernels)):
            self.seq.append(
                nn.ConvTranspose2d(
                    channels[i], channels[i + 1], kernel_size=kernels[i], stride=strides[i]
                )
            )
            if i < len(kernels) - 1:
                self.seq.append(Activation())

    def forward(self, x: Tensor):
        """"""
        x = self.first(x)
        x = x.reshape(-1, self.dim_middle, 1, 1)

        if self.debug:
            print(f"=== {self.__class__.__name__} ===")
            print(x.shape)
            for e in self.seq:
                x = e(x)
                print(x.shape)
        else:
            x = self.seq(x)

        return x


class SpatialBroadcastDecoder64(nn.Module):
    r"""
    (N, dim_input) -> (N, 3, 64, 64)

    References:
        Spatial Broadcast Decoder https://arxiv.org/abs/1901.07017

    References for implementation:
        https://github.com/dfdazac/vaesbd/blob/master/model.py#L6
    """

    def __init__(
        self,
        *,
        dim_input: int,
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
            nn.Conv2d(64, 3, 3, padding=1),
        )

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

        x = self.dec_convs(x)
        return x


class VisualEncoder256(VisualEncoder64):
    def __init__(
        self,
        dim_output: int,
        activation: str = "ReLU",
        channels: list = [3, 8, 16, 32, 64, 128, 256],
        kernels: list = [4, 5, 4, 4, 4, 4],
        strides: list = [2, 2, 2, 2, 2, 2],
        img_size: list = [256, 256],
        debug: bool = False,
    ) -> None:
        super().__init__(dim_output, activation, channels, kernels, strides, img_size, debug)


class VisualDecoder256(VisualDecoder64):
    def __init__(
        self,
        dim_input: int,
        dim_middle: int,
        activation: str = "ReLU",
        channels: list = [3, 8, 16, 32, 64, 128],
        kernels: list = [6, 6, 5, 5, 5, 5],
        strides: list = [2, 2, 2, 2, 2, 2],
        debug: bool = False,
    ):
        super().__init__(dim_input, dim_middle, activation, channels, kernels, strides, debug)


class VisualDecoder224(VisualDecoder64):
    def __init__(
        self,
        dim_input: int,
        dim_middle: int,
        activation: str = "ReLU",
        channels: list = [3, 8, 16, 32, 64, 128],
        kernels: list = [6, 6, 5, 5, 5, 4],
        strides: list = [2, 2, 2, 2, 2, 2],
        debug: bool = False,
    ):
        super().__init__(dim_input, dim_middle, activation, channels, kernels, strides, debug)


class ResNet(nn.Module):
    def __init__(
        self, dim_output: int, version: str = "resnet18", weights: Optional[str] = None
    ) -> None:
        """
        version: "resnet18", "resnet34", etc.
        weights: "IMAGENET1K_V1", etc.
        """

        super().__init__()

        self.m: models.ResNet = getattr(models, version)(weights=weights)
        self.m.fc = torch.nn.Linear(self.m.fc.in_features, dim_output)

    def forward(self, x: Tensor):
        return self.m(x)


class VisualDecoder224V2(nn.Module):
    def __init__(self, zsize):
        super().__init__()
        self.dfc3 = nn.Linear(zsize, 4096)
        self.bn3 = nn.BatchNorm2d(4096)
        self.dfc2 = nn.Linear(4096, 4096)
        self.bn2 = nn.BatchNorm2d(4096)
        self.dfc1 = nn.Linear(4096, 256 * 6 * 6)
        self.bn1 = nn.BatchNorm2d(256 * 6 * 6)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.dconv5 = nn.ConvTranspose2d(256, 256, 3, padding=0)
        self.dconv4 = nn.ConvTranspose2d(256, 384, 3, padding=1)
        self.dconv3 = nn.ConvTranspose2d(384, 192, 3, padding=1)
        self.dconv2 = nn.ConvTranspose2d(192, 64, 5, padding=2)
        self.dconv1 = nn.ConvTranspose2d(64, 3, 12, stride=4, padding=4)

    def forward(self, x: Tensor):  # ,i1,i2,i3):
        B = x.shape[0]

        x = self.dfc3(x)
        # x = F.relu(x)
        x = F.relu(self.bn3(x))

        x = self.dfc2(x)
        x = F.relu(self.bn2(x))
        # x = F.relu(x)
        x = self.dfc1(x)
        x = F.relu(self.bn1(x))
        # x = F.relu(x)
        # print(x.size())
        x = x.view(B, 256, 6, 6)
        # print (x.size())
        x = self.upsample1(x)
        # print x.size()
        x = self.dconv5(x)
        # print x.size()
        x = F.relu(x)
        # print x.size()
        x = F.relu(self.dconv4(x))
        # print x.size()
        x = F.relu(self.dconv3(x))
        # print x.size()
        x = self.upsample1(x)
        # print x.size()
        x = self.dconv2(x)
        # print x.size()
        x = F.relu(x)
        x = self.upsample1(x)
        # print x.size()
        x = self.dconv1(x)
        # print x.size()
        x = F.sigmoid(x)
        # print x
        return x


class ResNetDecoder(nn.Module):
    def __init__(
        self, dim_input: int, img_size: int = 224, Ch: List[int] = [3, 64, 128, 256, 512, 512]
    ) -> None:
        super().__init__()

        if img_size == 224:
            scale_factor = 7
        elif img_size == 256:
            scale_factor = 8
        else:
            assert False

        Ch = list(reversed(Ch))

        self.dim_middle = Ch[0]

        self.fc = nn.Linear(dim_input, self.dim_middle)
        self.upsample = nn.Upsample(scale_factor=scale_factor)

        blocks = []
        for i in range(len(Ch) - 1):
            blocks.append(from_stable_diffusion.SimpleDecoder(Ch[i], Ch[i + 1]))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: Tensor):
        x = self.fc(x)
        x = x.reshape(-1, self.dim_middle, 1, 1)
        x = self.upsample(x)
        x = self.blocks(x)
        return x


class DoubleConv(nn.Module):
    """
    same size as input

    References:
        https://github.com/milesial/Pytorch-UNet/blob/2f62e6b1c8e98022a6418d31a76f6abd800e5ae7/unet/unet_parts.py#L8
    """

    # ⌊(H + 2 * 1 - 1*(3 - 1) - 1)/1 +1⌋ = ⌊H⌋ = H

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        last: str = "ReLU",  # Identity, Sigmoid, ...
        bias: bool = True,
    ):

        if last == "ReLU":
            last_module = nn.ReLU(inplace=True)
        else:
            last_module = getattr(nn, last)

        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            last_module,
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv
    https://github.com/milesial/Pytorch-UNet/blob/2f62e6b1c8e98022a6418d31a76f6abd800e5ae7/unet/unet_parts.py#L28
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class EncoderV1(nn.Module):
    """
    References:
        https://github.com/ctallec/world-models/blob/d6abd9ce97409734a766eb67ccf0d1967ba9bf0c/models/vae.py#L32
    """

    def __init__(self, img_channels: int):
        super().__init__()

        self.conv1 = nn.Conv2d(img_channels, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)

        # self.conv1.requires_grad_

        # self.fc_mu = nn.Linear(2*2*256, latent_size)
        # self.fc_logsigma = nn.Linear(2*2*256, latent_size)

    def forward(self, x: Tensor):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)

        return x


class DecoderV1(nn.Module):
    """
    References:
        https://github.com/ctallec/world-models/blob/d6abd9ce97409734a766eb67ccf0d1967ba9bf0c/models/vae.py#L10
    """

    def __init__(self, img_channels: int):
        super().__init__()

        # self.fc1 = nn.Linear(latent_size, 1024)

        self.deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, img_channels, 6, stride=2)

    def forward(self, x: Tensor):
        # x = F.relu(self.fc1(x))
        # x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = self.deconv4(x)
        return x
