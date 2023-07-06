import itertools
import math
from numbers import Real
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import NumberType, Tensor, nn
from torch.nn.common_types import _size_2_t
from torch.nn.modules import activation as activations
from torch.nn.modules.utils import _pair
from torch.nn.utils import parametrizations  # .spectral_norm

from models import from_stable_diffusion, limiter
from models.mobile_unet import IRBDecoder
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
        in_channels: int = 3,
        dim_middle: int = 1024,
        activation: str = "ReLU",
        channels: list = None,
        kernels: list = None,
        strides: list = None,
        img_size: list = None,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()

        channels: list = channels or [32, 64, 128, 256]
        kernels: list = kernels or [4, 4, 4, 4]
        strides: list = strides or [2, 2, 2, 2]
        img_size: list = img_size or [64, 64]

        assert len(kernels) == len(channels)
        assert len(kernels) == len(strides)

        Activation = getattr(nn, activation)

        channels = [in_channels] + channels
        self.seq = nn.Sequential()
        # out -> z
        for i in range(len(kernels) - 1):
            if use_spectral_norm:
                wrap = parametrizations.spectral_norm
            else:
                wrap = lambda x: x

            self.seq.append(
                wrap(
                    nn.Conv2d(
                        channels[i],
                        channels[i + 1],
                        kernel_size=kernels[i],
                        stride=strides[i],
                    )
                )
            )
            self.seq.append(Activation())

        # last layer
        self.seq.append(
            nn.Conv2d(
                channels[-2],
                channels[-1],
                kernel_size=kernels[-1],
                stride=strides[-1],
            )
        )
        self.seq.append(Activation())

        self.last = nn.Identity() if dim_output == dim_middle else nn.Linear(dim_middle, dim_output)

    def forward(self, x: Tensor):
        """"""

        assert x.ndim == 4  # .unsqueeze(0)

        x = self.seq(x)  # (N, 256, 2, 2)
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
        out_channels: int = 3,
        dim_middle: int = 1024,
        activation: str = "ReLU",
        channels: list = None,
        kernels: list = None,
        strides: list = None,
        use_spectral_norm: bool = False,
    ):
        super().__init__()

        channels: list = channels or [32, 64, 128]
        kernels: list = kernels or [6, 6, 5, 5]
        strides: list = strides or [2, 2, 2, 2]

        assert len(kernels) == len(channels) + 1
        assert len(kernels) == len(strides)

        channels = [out_channels] + channels

        self.dim_middle = dim_middle
        self.first = nn.Linear(dim_input, dim_middle)
        Activation = getattr(nn, activation)

        channels.append(dim_middle)
        channels = list(reversed(channels))
        kernels = list(reversed(kernels))
        strides = list(reversed(strides))
        self.seq = nn.Sequential()
        # z -> out
        for i in range(len(kernels) - 1):
            if use_spectral_norm:
                wrap = parametrizations.spectral_norm
            else:
                wrap = lambda x: x

            self.seq.append(
                wrap(
                    nn.ConvTranspose2d(
                        channels[i], channels[i + 1], kernel_size=kernels[i], stride=strides[i]
                    )
                )
            )
            self.seq.append(Activation())

        # last layer
        self.seq.append(
            nn.ConvTranspose2d(
                channels[-2],
                channels[-1],
                kernel_size=kernels[-1],
                stride=strides[-1],
            )
        )

    def forward(self, x: Tensor):
        """"""
        x = self.first(x)
        x = x.reshape(-1, self.dim_middle, 1, 1)
        x = self.seq(x)
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
        self,
        dim_output: int,
        version: str = "resnet18",
        weights: Optional[str] = None,
        in_channels: int = 3,
    ) -> None:
        """
        version: "resnet18", "resnet34", etc.
        weights: "IMAGENET1K_V1", etc.

        Reference:
            https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html
            https://pytorch.org/vision/main/models/generated/torchvision.models.resnet34.html
            ...
        """

        super().__init__()

        self.m: torchvision.models.ResNet = getattr(torchvision.models, version)(weights=weights)
        if in_channels != 3:
            self.m.conv1 = nn.Conv2d(
                in_channels,
                out_channels=self.m.conv1.out_channels,
                kernel_size=self.m.conv1.kernel_size,  # 7
                stride=self.m.conv1.stride,  # 2
                padding=self.m.conv1.padding,  # 3
                bias=False,
            )
        self.m.fc = torch.nn.Linear(self.m.fc.in_features, dim_output)

    def forward(self, x: Tensor):
        return self.m(x)


# class ResNetBackBone(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         # AdaptiveAvgPool2d はパラメーターを持っていない　すなわちただの関数
#         # out = torch.tanh(out)


class MLP(torch.nn.Sequential):
    """
    Linear
    NormLayer
    Activation
    Linear
    NormLayer
    Activation
    ...
    Linear

    References:
        - torchvision.ops.MLP
        - https://github.com/lucidrains/vit-pytorch/blob/ce4bcd08fbab864e92167415552a722ff5ce2005/vit_pytorch/es_vit.py#L118
    """

    def __init__(
        self,
        dim_in: int,
        dim_hiddens: List[int],  # same as number of Linear layer
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
        activation: Union[str, Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        inplace: Optional[bool] = True,
        bias: bool = True,
    ):
        # The addition of `norm_layer` is inspired from the implementation of TorchMultimodal:
        # https://github.com/facebookresearch/multimodal/blob/5dec8a/torchmultimodal/modules/layers/mlp.py
        params = {} if inplace is None else {"inplace": inplace}

        if type(activation) == str:
            activation = getattr(nn, activation)

        layers = []
        in_dim = dim_in
        for hidden_dim in dim_hiddens[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation(**params))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, dim_hiddens[-1], bias=bias))

        super().__init__(*layers)


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


class VisualDecoder224V3(nn.Module):
    """By ChatGPT"""

    def __init__(
        self,
        dim_input: int,
        activation: str = "ReLU",
        out_channels: int = 3,
    ):
        super().__init__()

        self.latent_dim = dim_input
        Activation = getattr(nn, activation)

        self.fc = nn.Linear(dim_input, 256 * 7 * 7)
        self.conv_transpose = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            Activation(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            Activation(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            Activation(inplace=True),
            nn.ConvTranspose2d(
                32, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.reshape(-1, 256, 7, 7)
        x = self.conv_transpose(x)
        x = nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return x


# class ResNetDecoder(nn.Module):
#     def __init__(
#         self, dim_input: int, img_size: int = 224, Ch: List[int] = [3, 64, 128, 256, 512, 512]
#     ) -> None:
#         super().__init__()

#         if img_size == 224:
#             scale_factor = 7
#         elif img_size == 256:
#             scale_factor = 8
#         else:
#             assert False

#         Ch = list(reversed(Ch))

#         self.dim_middle = Ch[0]

#         self.fc = nn.Linear(dim_input, self.dim_middle)
#         self.upsample = nn.Upsample(scale_factor=scale_factor)

#         blocks = []
#         for i in range(len(Ch) - 1):
#             blocks.append(from_stable_diffusion.SimpleDecoder(Ch[i], Ch[i + 1]))
#         self.blocks = nn.Sequential(*blocks)

#     def forward(self, x: Tensor):
#         x = self.fc(x)
#         x = x.reshape(-1, self.dim_middle, 1, 1)
#         x = self.upsample(x)
#         x = self.blocks(x)
#         return x


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
    r"""
    Input:  (N, 3, 64, 64)
    Output: (N, OC)

    OC = dim_output

    References:
        https://github.com/ctallec/world-models/blob/d6abd9ce97409734a766eb67ccf0d1967ba9bf0c/models/vae.py#L32
    """

    def __init__(self, dim_output: int, img_channels: int, freeze: str = "none"):
        super().__init__()

        assert freeze in ("none", "conv", "all")

        self.conv1 = nn.Conv2d(img_channels, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)

        self.fc = nn.Linear(2 * 2 * 256, dim_output)

        def frozen_conv():
            for m in [self.conv1, self.conv2, self.conv3, self.conv4]:
                m.requires_grad_(False)

        if freeze == "conv":
            frozen_conv()

        elif freeze == "all":
            for m in self.parameters():
                m.requires_grad_(False)

    def forward(self, x: Tensor):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class DecoderV1(nn.Module):
    r"""
    Input:  (N, IC)
    Output: (N, 3, 64, 64)

    IC = dim_input

    References:
        https://github.com/ctallec/world-models/blob/d6abd9ce97409734a766eb67ccf0d1967ba9bf0c/models/vae.py#L10
    """

    def __init__(self, dim_input: int, img_channels: int, freeze: str = "none"):
        super().__init__()

        assert freeze in ("none", "conv", "all")

        self.fc = nn.Linear(dim_input, 1024)

        self.deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, img_channels, 6, stride=2)

        def frozen_conv():
            for m in [self.deconv1, self.deconv2, self.deconv3, self.deconv4]:
                m.requires_grad_(False)

        if freeze == "conv":
            frozen_conv()

        elif freeze == "all":
            for m in self.parameters():
                m.requires_grad_(False)

    def forward(self, x: Tensor):
        x = F.relu(self.fc(x))
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = self.deconv4(x)
        return x


class DecoderV2(nn.Module):
    """
    Input:  (N, IC)
    Output: (N, OC, 64, 64)

    IC = dim_input
    OC = out_channels

    Reference:
        https://github.com/AntixK/PyTorch-VAE/blob/a6896b944c918dd7030e7d795a8c13e5c6345ec7/models/vanilla_vae.py#L40
    """

    def __init__(self) -> None:
        super().__init__()

    def __init__(
        self,
        dim_input: int,
        out_channels: int = 3,
        channels: List[int] = None,
    ) -> None:
        super().__init__()

        if channels is None:
            channels = [32, 64, 128, 256, 512]

        self.decoder_input = nn.Linear(dim_input, channels[-1] * 4)

        channels.reverse()
        self.decoder = nn.Sequential()
        for i in range(len(channels) - 1):
            self.decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        channels[i],
                        channels[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(channels[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                channels[-1],
                channels[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(channels[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(channels[-1], out_channels=out_channels, kernel_size=3, padding=1),
        )

    def forward(self, z: Tensor) -> Tensor:
        x = self.decoder_input(z)
        x = x.view(-1, 512, 2, 2)
        x = self.decoder(x)
        x = self.final_layer(x)
        return x


class IRBDecoderWrap(nn.Module):
    """
    Input:  (N, IC)
    Output: (N, OC, H, W)

    IC = dim_input
    OC = out_channels
    (H, W) = img_size
    """

    def __init__(
        self, *, dim_input: int, img_size: Union[int, tuple], out_channels: int = 3, C: int = 512
    ) -> None:
        """
        img_size : (H, W) or H or W
        """

        super().__init__()

        if type(img_size) == int:
            img_size = (img_size, img_size)

        assert img_size[0] % 32 == 0
        assert img_size[1] % 32 == 0

        self.zh = img_size[0] // 32
        self.zw = img_size[1] // 32
        self.C = C

        self.fc = nn.Linear(dim_input, C * self.zh * self.zw)
        self.core = IRBDecoder(out_c=out_channels, in_c=C)

    def forward(self, z: Tensor):
        x = self.fc(z)
        x = x.view(-1, self.C, self.zh, self.zw)
        x = self.core(x)
        return x


class DecoderV3(nn.Module):
    """
    Input:  (N, IC)
    Output: (N, OC, H, W)

    IC = dim_input
    OC = out_channels
    (H, W) = img_size

    Reference:
        https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(
        self,
        *,
        dim_input: int,
        img_size: Union[int, tuple],
        out_channels: int = 3,
        Ch: Optional[List[int]] = None,
    ) -> None:
        """
        img_size : (H, W)
            Must be a multiple of 32.
            If int type is specified, it is assumed to be a square image.
        """

        super().__init__()

        if type(img_size) == int:
            img_size = (img_size, img_size)

        assert img_size[0] % 32 == 0
        assert img_size[1] % 32 == 0

        self.zh = img_size[0] // 32
        self.zw = img_size[1] // 32

        if Ch is None:
            Ch = [512, 128, 64, 32, 16]

        assert len(Ch) == 5

        Ch.append(out_channels)
        self.C = Ch[0]
        self.fc = nn.Linear(dim_input, self.C * self.zh * self.zw)  # 振り返ればこの実装マズイなぁ　でも相関出る

        self.core = nn.Sequential()
        for i in range(5):
            self.core.append(
                nn.Sequential(
                    nn.BatchNorm2d(Ch[i]),
                    nn.ReLU(inplace=True),
                    from_stable_diffusion.Upsample(Ch[i], Ch[i + 1]),
                )
            )

    def forward(self, z: Tensor):
        x = self.fc(z)
        x = x.view(-1, self.C, self.zh, self.zw)
        x = self.core(x)
        return x


class DecoderC_(nn.Module):
    """
    Input:  (N, ic, ih, iw)
    Output: (N, oc, 32*ih, 32*iw)

    References:
        https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/diffusionmodules/model.py#L571
    """

    def __init__(
        self,
        *,
        in_c: int,
        out_c: int,
        middle_ch: Optional[List[int]] = None,
        residual="v2",
        use_spectral_norm=False,
    ) -> None:
        super().__init__()

        if middle_ch is None:
            middle_ch = [128, 64, 32, 16]

        Ch = [in_c] + middle_ch + [out_c]

        self.m = nn.Sequential()

        # z-> 0 1 ⭐2 3 4 -> img
        # https://www.researchgate.net/figure/Conceptual-overview-of-the-ResNet-building-block-and-the-ResNet-152-architecture-43_fig7_341576780
        for i in range(5):
            self.m.append(
                from_stable_diffusion.SimpleDecoder(
                    Ch[i], Ch[i + 1], residual=residual, use_spectral_norm=use_spectral_norm
                )
            )

    def forward(self, z: Tensor):
        return self.m(z)


class DecoderC(nn.Module):
    """
    Input:  (N, in_c)
    Output: (N, out_c, oh, ow)

    img_size : (oh, ow)
    oh and ow must be multiples of 32.

    ResNetの
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    で (N, OC(=512), 1, 1) の出力
    この 1, 1 を崩さないイメージ
    OCの各次元の独立状態を保つ

    まぁ多分あんま意味ない　が、Linearの行列サイズが小さいため計算効率が良い
    """

    def __init__(
        self,
        *,
        img_size: Union[int, tuple],
        out_c: int = 3,
        in_c=512,
        residual="v2",
        use_spectral_norm=False,
    ) -> None:
        super().__init__()

        if type(img_size) == int:
            img_size = (img_size, img_size)

        assert img_size[0] % 32 == 0
        assert img_size[1] % 32 == 0

        self.in_c = in_c
        self.zh = img_size[0] // 32
        self.zw = img_size[1] // 32

        middle = int(self.zh * self.zw // 2)
        self.fc1 = nn.Linear(1, middle)
        self.fc2 = nn.Linear(middle, self.zh * self.zw)
        self.core = DecoderC_(
            in_c=in_c, out_c=out_c, residual=residual, use_spectral_norm=use_spectral_norm
        )

    def forward(self, z: Tensor):
        x = z.reshape(-1, self.in_c, 1)
        x = self.fc1(x)
        x = F.mish(x)
        x = self.fc2(x)
        x = x.reshape(-1, self.in_c, self.zh, self.zw)
        x = self.core(x)
        return x


class InfratorV1(nn.Module):
    """
    Input:  (*)
    Output: (*, *out_size)
    """

    def __init__(
        self,
        out_size: Iterable[int],
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.out_size = out_size
        prod = math.prod(out_size)
        self.fc = nn.Linear(1, prod, bias=bias)

    def forward(self, x: Tensor):
        s_ = x.shape
        x = x.unsqueeze(-1)
        x = self.fc(x)
        x = x.reshape(*s_, *self.out_size)
        return x


class InfratorV2(nn.Module):
    """
    Input:  (*)
    Output: (*, *out_size)
    """

    # for jump source code
    # torchvision.ops.MLP
    # nn.Conv2d
    # nn.SiLU # swish
    # nn.Mish
    # nn.Identity # not in torch.nn.modules.activation

    def __init__(
        self,
        out_size: Iterable[int],
        activation: Union[str, nn.Module] = nn.Mish,
        inplace: Optional[bool] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        params = {} if inplace is None else {"inplace": inplace}

        self.out_size = out_size
        prod = math.prod(out_size)
        middle = prod // 2
        middle = int(math.sqrt(middle))
        self.fc1 = nn.Linear(1, middle, bias=bias)
        self.act_fn = (activation if type(activation) != str else getattr(nn, activation))(**params)
        self.fc2 = nn.Linear(middle, prod, bias=bias)

    def forward(self, x: Tensor):
        s_ = x.shape
        x = x.unsqueeze(-1)
        x = self.fc2(self.act_fn(self.fc1(x)))
        x = x.reshape(*s_, *self.out_size)
        return x


class ResNetC(nn.Module):
    """
    Input:  (N, 3, Any, Any)
    Output: (N, 512)

    nn.Linear does not exist
    """

    def __init__(self, version: str = "resnet18", weights: Optional[str] = None) -> None:
        """
        version: "resnet18", "resnet34", etc.
        weights: "IMAGENET1K_V1", etc.
        """

        super().__init__()

        # https://github.com/pytorch/vision/blob/5b07d6c9c6c14cf88fc545415d63021456874744/torchvision/models/resnet.py#L166
        self.m: torchvision.models.ResNet = getattr(torchvision.models, version)(weights=weights)
        self.m.fc = nn.Identity()

    def forward(self, x: Tensor):
        return self.m(x)


class DecoderCWrap(nn.Module):
    def __init__(self, dim_input: int, in_c=512, last_fn=None, **kwargs) -> None:
        super().__init__()
        self.fc = nn.Linear(dim_input, in_c)
        self.core = DecoderC(in_c=in_c, **kwargs)

        # self.core.load_state_dict()
        # self.core.requires_grad_()

        self.last_fn = getattr(limiter, last_fn, lambda x: x)

    def forward(self, x: Tensor):
        x = self.fc(x)
        x = self.core(x)
        x = self.last_fn(x)
        return x


class ResnetDecoder(nn.Module):
    """
    Input:  (N, dim_input)
    Output: (N, out_channels, oh, ow)

    img_size : (oh, ow)
    oh and ow must be multiples of 32.
    """

    def __init__(
        self,
        dim_input: int,
        img_size: Union[int, tuple],
        out_channels: int = 3,
        middle_ch: Optional[List[int]] = None,
        residual="v2",
        use_spectral_norm=False,
        last_fn=None,
    ) -> None:
        super().__init__()

        if type(img_size) == int:
            img_size = (img_size, img_size)

        assert img_size[0] % 32 == 0
        assert img_size[1] % 32 == 0

        self.zh = img_size[0] // 32
        self.zw = img_size[1] // 32

        if middle_ch is None:
            middle_ch = [256, 128, 64, 32, 16]

        self.ic = middle_ch[0]

        Ch = middle_ch + [out_channels]

        O = self.ic * self.zh * self.zw
        mid = int(O // 2)
        self.fc1 = nn.Linear(dim_input, mid)
        self.fc2 = nn.Linear(mid, O)
        # self.fc = nn.Linear(dim_input, O)

        self.m = nn.Sequential()
        # z-> 0 1 ⭐2 3 4 -> img
        # https://www.researchgate.net/figure/Conceptual-overview-of-the-ResNet-building-block-and-the-ResNet-152-architecture-43_fig7_341576780
        for i in range(5):
            self.m.append(
                from_stable_diffusion.SimpleDecoder(
                    Ch[i], Ch[i + 1], residual=residual, use_spectral_norm=use_spectral_norm
                )
            )

        self.last_fn = getattr(limiter, last_fn, lambda x: x)

    def forward(self, x: Tensor):
        x = self.fc1(x)
        x = F.mish(x)
        x = self.fc2(x)
        x = x.reshape(-1, self.ic, self.zh, self.zw)
        x = self.m(x)
        x = self.last_fn(x)
        return x


class ResnetCWrap(nn.Module):
    def __init__(self, dim_output: int, **kwargs) -> None:
        super().__init__()
        self.core = ResNetC(**kwargs)
        self.fc = nn.Linear(512, dim_output)

        # self.core.load_state_dict()
        # self.core.requires_grad_()

    def forward(self, x: Tensor):
        x = self.core(x)
        x = self.fc(x)
        return x


# class SimpleAttentionNetwork(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()

#         base_model = resnet34(pretrained=True)
#         self.features = nn.Sequential(*[layer for layer in base_model.children()][:-2])
#         self.attn_conv = nn.Sequential(nn.Conv2d(512, 1, 1), nn.Sigmoid())
#         self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, num_classes))
#         self.mask_ = None

#     def forward(self, x):
#         x = self.features(x)

#         attn = self.attn_conv(x)  # [B, 1, H, W]
#         B, _, H, W = attn.shape
#         self.mask_ = attn.detach().cpu()

#         x = x * attn
#         x = F.adaptive_avg_pool2d(x, (1, 1))
#         x = x.reshape(B, -1)

#         return self.fc(x)

# def save_attention_mask(self, x, path):
#     B = x.shape[0]
#     self.forward(x)
#     x = x.cpu() * torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
#     x = x + torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
#     fig, axs = plt.subplots(4, 2, figsize=(6, 8))
#     plt.axis("off")
#     for i in range(4):
#         axs[i, 0].imshow(x[i].permute(1, 2, 0))
#         axs[i, 1].imshow(self.mask_[i][0])
#     plt.savefig(path)
#     plt.close()


# class SelfAttention(nn.Module):
#     """
#     https://discuss.pytorch.org/t/attention-in-image-classification/80147
#     """

#     def __init__(self, in_channels):
#         super().__init__()

#         self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
#         self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
#         self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

#         self.gamma = nn.Parameter(torch.zeros(1))

#     def forward(self, x):
#         B, C, H, W = x.size()

#         proj_query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)
#         proj_key = self.key_conv(x).view(B, -1, H * W)

#         energy = torch.bmm(proj_query, proj_key)
#         attention = torch.softmax(energy, dim=-1)

#         proj_value = self.value_conv(x).view(B, -1, H * W)

#         out = torch.bmm(proj_value, attention.permute(0, 2, 1))
#         out = out.view(B, C, H, W)

#         out = self.gamma * out + x

#         return out, attention


def positional_encoding_item(d_model: int, max_len: int):
    """
    Returns:
        pe: (max_len, d_model)

    Attention Is All You Need
    PE(pos, 2i) = sin(pos/100002i/dmodel)
    PE(pos, 2i+1) = cos(pos/100002i/dmodel)

    https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/
    https://erdem.pl/2021/05/understanding-positional-encoding-in-transformers
    https://discuss.pytorch.org/t/transformer-example-position-encoding-function-works-only-for-even-d-model/100986/2
    https://datascience.stackexchange.com/questions/82451/why-is-10000-used-as-the-denominator-in-positional-encodings-in-the-transformer
    """
    # nn.Embedding

    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    if d_model % 2 == 0:
        pe[:, 1::2] = torch.cos(position * div_term)
    else:
        pe[:, 1::2] = torch.cos(position * div_term[:-1])
    return pe


def positional_encoding_func(d_model: int, t: int):
    """
    Returns:
        pe: (d_model)

    Attention Is All You Need
    PE(pos, 2i) = sin(pos/100002i/dmodel)
    PE(pos, 2i+1) = cos(pos/100002i/dmodel)

    https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/
    https://erdem.pl/2021/05/understanding-positional-encoding-in-transformers
    https://discuss.pytorch.org/t/transformer-example-position-encoding-function-works-only-for-even-d-model/100986/2
    https://datascience.stackexchange.com/questions/82451/why-is-10000-used-as-the-denominator-in-positional-encodings-in-the-transformer
    """
    # nn.Embedding

    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(d_model)
    pe[0::2] = torch.sin(t * div_term)
    if d_model % 2 == 0:
        pe[1::2] = torch.cos(t * div_term)
    else:
        pe[1::2] = torch.cos(t * div_term[:-1])
    return pe
