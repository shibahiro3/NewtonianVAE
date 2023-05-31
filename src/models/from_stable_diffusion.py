"""
x2
7 14 28 56 112 224
8 16 32 64 128 256

without ConvTranspose

References:
    https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/model.py
    https://github.com/Stability-AI/stablediffusion/blob/main/ldm/modules/diffusionmodules/model.py
"""


from functools import partial, singledispatch
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils import parametrizations  # .spectral_norm

from mypython.terminal import Color


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    # return torch.nn.GroupNorm(
    #     num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    # )
    # num_channels must be divisible by num_groups
    # 16とか3chに使えない　そもそもBatchNormの方が良いらしい　計算コストの話
    # resnetが3ch以外使え無さそうなのもこれが原因
    return nn.BatchNorm2d(in_channels)


# Default Conv2d:
# kernel_size: must needed
# stride: 1
# padding: 0
# -> convX10


def conv311(IC: int, OC: int):
    """
    conv3x3

    same size as input

    References:
        https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/diffusionmodules/model.py#L47
        https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/diffusionmodules/model.py#L92
    """
    return torch.nn.Conv2d(IC, OC, kernel_size=3, stride=1, padding=1)


def conv320(IC: int, OC: int):
    """
    conv3x3

    Input  (*, H,            W)
    Output (*, ⌊(H-3)/2 +1⌋, ⌊(W-3)/2 +1⌋)

    References:
        https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/diffusionmodules/model.py#L66
    """
    return torch.nn.Conv2d(IC, OC, kernel_size=3, stride=2, padding=0)


def conv110(IC: int, OC: int):
    """
    same size as input

    References:
        https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/diffusionmodules/model.py#L115
        ...
    """
    return nn.Conv2d(IC, OC, kernel_size=1, stride=1, padding=0)


class Upsample(nn.Module):
    """
    Input  (*, H,  W)
    Output (*, 2H, 2W)

    with_conv can be either

    References:
        https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/diffusionmodules/model.py#L42
    """

    def __init__(self, IC: int, OC: int, with_conv: bool = True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = conv311(IC, OC)

    def forward(self, x: Tensor):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    Input  (*, H,     W)
    Output (*, ⌊H/2⌋, ⌊W/2⌋)

    References:
        https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/diffusionmodules/model.py#L60
    """

    def __init__(self, in_channels: int, with_conv: bool):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = conv320(in_channels, in_channels)

    def forward(self, x: Tensor):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    """
    same size as input?

    https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/diffusionmodules/model.py#L82
    """

    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
        use_spectral_norm=False
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        if use_spectral_norm:
            wrap = parametrizations.spectral_norm
        else:
            wrap = lambda x: x

        self.norm1 = Normalize(in_channels)
        self.conv1 = wrap(conv311(in_channels, out_channels))
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = wrap(conv311(out_channels, out_channels))
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = wrap(conv311(in_channels, out_channels))
            else:
                self.nin_shortcut = wrap(conv110(in_channels, out_channels))

    def forward(self, x: Tensor, temb: Optional[Tensor]):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class SimpleDecoder(nn.Module):
    """
    Input  (*, H,  W)
    Output (*, 2H, 2W)

    References:
        https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/diffusionmodules/model.py#L571
        https://www.researchgate.net/figure/Conceptual-overview-of-the-ResNet-building-block-and-the-ResNet-152-architecture-43_fig7_341576780
    """

    def __init__(self, in_channels, out_channels, residual="v1", use_spectral_norm=False):
        super().__init__()

        assert residual in ("none", "v1", "v2", "v3")

        if use_spectral_norm:
            wrap = parametrizations.spectral_norm
        else:
            wrap = lambda x: x

        model = []

        if residual == "v1":
            # same original impl
            model += [
                wrap(conv110(in_channels, in_channels)),
                ResnetBlock(
                    in_channels=in_channels,
                    out_channels=2 * in_channels,
                    temb_channels=0,
                    dropout=0.0,
                    use_spectral_norm=use_spectral_norm,
                ),
                ResnetBlock(
                    in_channels=2 * in_channels,
                    out_channels=4 * in_channels,
                    temb_channels=0,
                    dropout=0.0,
                    use_spectral_norm=use_spectral_norm,
                ),
                ResnetBlock(
                    in_channels=4 * in_channels,
                    out_channels=2 * in_channels,
                    temb_channels=0,
                    dropout=0.0,
                    use_spectral_norm=use_spectral_norm,
                ),
                wrap(conv110(2 * in_channels, in_channels)),
            ]

        elif residual == "v2":
            model += [
                wrap(conv110(in_channels, in_channels)),
                ResnetBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=0,
                    dropout=0.0,
                    use_spectral_norm=use_spectral_norm,
                ),
                wrap(conv110(in_channels, in_channels)),
            ]

        elif residual == "v3":
            model += [
                ResnetBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=0,
                    dropout=0.0,
                    use_spectral_norm=use_spectral_norm,
                ),
            ]

        model += [Upsample(in_channels, in_channels, with_conv=True)]

        self.model = nn.ModuleList(model)

        # end
        self.norm_out = Normalize(in_channels)
        self.conv_out = conv311(in_channels, out_channels)  # no wrap
        # self.conv_out = wrap(conv311(in_channels, out_channels))

    def forward(self, x: Tensor):
        for layer in self.model:
            if type(layer) == ResnetBlock:
                x = layer(x, None)
            else:
                x = layer(x)

        h = self.norm_out(x)
        h = nonlinearity(h)
        x = self.conv_out(h)
        return x
