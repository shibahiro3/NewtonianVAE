"""
https://pytorch.org/docs/stable/nn.html
"""


import math
from functools import singledispatch
from numbers import Real
from pprint import pprint

import torch.nn.functional as F
from torch import nn

from mypython.terminal import Color


@singledispatch
def get_size(model, size: tuple) -> tuple:
    """
    same size as input
        https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
        https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html
        ...
    """
    return size


def last_size(seq, size: tuple):
    for model in seq:
        in_size = size
        size = get_size(model, size)
        out_size = size

        # if type(model) == nn.Conv2d:
        #     print(
        #         in_size,
        #         "<->",
        #         get_size_conv_transpose2d(copy_conv_hparam(model, swap_io=True), out_size),
        #     )

    return size


########## Convolution Layers ##########
# https://pytorch.org/docs/stable/nn.html#convolution-layers


def _conv(in_, kernel_size, stride=1, padding=0, dilation=1):
    return math.floor((in_ + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)


@get_size.register
def get_size_conv2d(model: nn.Conv2d, size: tuple) -> tuple:
    """
    https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    Input:  (N, C_in,  H_in,  W_in),  (C_in,  H_in, W_in)
    Output: (N, C_out, H_out, W_out), (C_out, H_out, W_out)
    """
    # F.conv2d

    assert size[-3] == model.in_channels
    assert model.in_channels % model.groups == 0
    assert model.out_channels % model.groups == 0

    C_out = model.out_channels
    H_out = _conv(
        size[-2],
        kernel_size=model.kernel_size[0],
        stride=model.stride[0],
        padding=model.padding[0],
        dilation=model.dilation[0],
    )
    W_out = _conv(
        size[-1],
        kernel_size=model.kernel_size[1],
        stride=model.stride[1],
        padding=model.padding[1],
        dilation=model.dilation[1],
    )

    # if model.groups == model.in_channels and (model.out_channels % model.in_channels == 0):
    #     print("depthwise convolution")

    if len(size) <= 3:
        return (C_out, H_out, W_out)
    else:
        return size[:-3] + (C_out, H_out, W_out)


def _conv_transpose(out, kernel_size, stride=1, padding=0, dilation=1, output_padding=0):
    return (out - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1


class copy_conv_hparam:
    def __init__(self, model, swap_io=False) -> None:
        self.in_channels = model.in_channels
        self.out_channels = model.out_channels

        if swap_io:
            self.swap_io()

        self.kernel_size = model.kernel_size
        self.stride = model.stride
        self.padding = model.padding
        self.dilation = model.dilation
        self.output_padding = model.output_padding
        self.groups = model.groups

    def swap_io(self):
        in_ = self.in_channels
        self.in_channels = self.out_channels
        self.out_channels = in_


@get_size.register
def get_size_conv_transpose2d(model: nn.ConvTranspose2d, size: tuple) -> tuple:
    """
    https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
    Input:  (N, C_in,  H_in,  W_in),  (C_in,  H_in,  W_in)
    Output: (N, C_out, H_out, W_out), (C_out, H_out, W_out)
    """
    # F.conv_transpose2d

    assert size[-3] == model.in_channels
    assert model.in_channels % model.groups == 0
    assert model.out_channels % model.groups == 0

    C_out = model.out_channels
    H_out = _conv_transpose(
        out=size[-2],
        kernel_size=model.kernel_size[0],
        stride=model.stride[0],
        padding=model.padding[0],
        dilation=model.dilation[0],
        output_padding=model.output_padding[0],
    )
    W_out = _conv_transpose(
        out=size[-1],
        kernel_size=model.kernel_size[1],
        stride=model.stride[1],
        padding=model.padding[1],
        dilation=model.dilation[1],
        output_padding=model.output_padding[1],
    )

    if len(size) <= 3:
        return (C_out, H_out, W_out)
    else:
        return size[:-3] + (C_out, H_out, W_out)


########## Pooling layers ##########
# https://pytorch.org/docs/stable/nn.html#pooling-layers


def _max_unpool2d(in_, kernel_size, stride, padding=0):
    return (in_ - 1) * stride - 2 * padding + kernel_size


@get_size.register
def get_size_max_unpool2d(model: nn.MaxUnpool2d, size: tuple) -> tuple:
    """
    https://pytorch.org/docs/stable/generated/torch.nn.MaxUnpool2d.html
    Input:  (N, C, H_in,  W_in),  (C, H_in, W_in)
    Output: (N, C, H_out, W_out), (C, H_out, W_out)
    """
    # F.max_unpool2d

    H_out = _max_unpool2d(
        in_=size[-2],
        kernel_size=model.kernel_size[0],
        stride=model.stride[0],
        padding=model.padding[0],
    )
    W_out = _max_unpool2d(
        in_=size[-1],
        kernel_size=model.kernel_size[1],
        stride=model.stride[1],
        padding=model.padding[1],
    )

    return size[:-2] + (H_out, W_out)


def _avg_pool(in_, kernel_size, stride, padding, ceil_mode):
    s = in_ + 2 * padding - kernel_size / stride + 1
    if ceil_mode:
        return math.ceil(s)
    else:
        return math.floor(s)


@get_size.register
def get_size_avg_pool2d(model: nn.AvgPool2d, size: tuple) -> tuple:
    """
    https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html
    Input:  (N, C, H_in,  W_in),  (C, H_in,  W_in)
    Output: (N, C, H_out, W_out), (C, H_out, W_out)
    """
    # F.avg_pool2d

    H_out = _avg_pool(
        in_=size[-2],
        kernel_size=model.kernel_size[0],
        stride=model.stride[0],
        padding=model.padding[0],
        ceil_mode=model.ceil_mode,
    )
    W_out = _avg_pool(
        in_=size[-1],
        kernel_size=model.kernel_size[1],
        stride=model.stride[1],
        padding=model.padding[1],
        ceil_mode=model.ceil_mode,
    )

    return size[:-2] + (H_out, W_out)


def _adaptive_avg_pool(size, output_size):
    return output_size if output_size is not None else size


@get_size.register
def get_size_adaptive_avg_pool2d(model: nn.AdaptiveAvgPool2d, size: tuple) -> tuple:
    """
    https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html
    Input:  (N, C, H_in, W_in), (C, H_in, W_in)
    Output: (N, C, S_0,  S_1),  (C, S_0,  W_1)
    """
    # F.adaptive_avg_pool2d

    S_0 = _adaptive_avg_pool(size=size[-2], output_size=model.output_size[-2])
    S_1 = _adaptive_avg_pool(size=size[-1], output_size=model.output_size[-1])

    return size[:-2] + (S_0, S_1)


########## Linear Layers ##########
# https://pytorch.org/docs/stable/nn.html#linear-layers


@get_size.register
def get_size_linear(model: nn.Linear, size: tuple) -> tuple:
    """
    https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear
    Input:  (*, H_in)
    Output: (*, H_out)
    """
    # F.linear

    assert size[-1] == model.in_features

    return size[:-1] + (model.out_features,)


########## Vision Layers ##########
# https://pytorch.org/docs/stable/nn.html#vision-layers


def _upsample(size, scale_factor):
    scale_factor = scale_factor if isinstance(scale_factor, Real) else scale_factor[0]
    return math.floor(size * scale_factor)


@get_size.register
def get_size_upsample(model: nn.Upsample, size: tuple) -> tuple:
    """
    https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html
    Input:  (N, C, W_in),  (N, C, H_in,  W_in),  (N, C, D_in,  H_in, W_in)
    Output: (N, C, W_out), (N, C, H_out, W_out), (N, C, D_out, H_out, W_out)
    """
    # F.upsample
    # F.interpolate

    if len(size) == 3:
        W_out = _upsample(size[-1], model.scale_factor)
        return size[:-1] + (W_out,)
    elif len(size) == 4:
        H_out = _upsample(size[-2], model.scale_factor[0])
        W_out = _upsample(size[-1], model.scale_factor[1])
        return size[:-2] + (H_out, W_out)
    elif len(size) == 5:
        D_out = _upsample(size[-3], model.scale_factor[0])
        H_out = _upsample(size[-2], model.scale_factor[1])
        W_out = _upsample(size[-1], model.scale_factor[2])
        return size[:-3] + (D_out, H_out, W_out)
