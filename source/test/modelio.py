import os
import sys

sys.path.append((os.pardir + os.sep) * 1)

from functools import singledispatch, singledispatchmethod

import numpy as np
import torch
import torchvision.models as models
from torchsummary import summary

import models.resnet_decoder as rd
import models.resnet_original as ro
from models import encdec, resnet2
from mypython.ai import nnio

# from source.models.resnet import BasicBlock, Bottleneck
from mypython.terminal import Color

B = 7
dim_latent = 13


def img64():
    print("=== img64 ===")

    img_size = (B, 3, 64, 64)
    img = torch.randn(img_size)

    enc = encdec.VisualEncoder64(dim_latent, debug=True)
    z = enc(img)
    print("z:", z.shape)
    assert z.shape == (B, dim_latent)

    dec = encdec.VisualDecoder64(dim_latent, dim_middle=1234, debug=True)
    dec.debug = True
    x = dec(z)
    print("x:", x.shape)
    assert x.shape == img_size


def img256():
    print("=== img256 ===")

    img_size = (B, 3, 256, 256)
    img = torch.randn(img_size)

    enc = encdec.VisualEncoder256(dim_latent, debug=True)
    z = enc(img)
    print("z:", z.shape)
    assert z.shape == (B, dim_latent)

    dec = encdec.VisualDecoder256(dim_latent, dim_middle=1234, debug=True)
    x = dec(z)
    print("x:", x.shape)
    assert x.shape == img_size


def resnet18():
    print("=== resnet18 ===")

    img = torch.randn(B, 3, 224, 224)

    enc = encdec.ResNet(dim_latent)
    summary(enc, img.shape[-3:], device="cpu")

    # enc = encdec.ResNet(dim_latent, version="resnet50", weights="IMAGENET1K_V1")
    z = enc(img)
    print("z:", z.shape)
    assert z.shape == (B, dim_latent)

    dec = encdec.VisualDecoder224(dim_latent, dim_middle=1234, debug=True)
    x = dec(z)
    print("x:", x.shape)
    assert x.shape == img.shape

    # print(enc)
    # print(dec)


def decoderV2():

    # resnet50
    z = torch.randn((B, 48))
    dec = encdec.VisualDecoder224V2(48)
    print(dec)
    x = dec(z)
    print(x.shape)

    # resnet18
    # z = torch.randn((B, dim_latent))
    # dec = encdec.ResNetDecoder(BasicBlock, [2, 2, 2, 2])


def res_orig():
    # models.resnet18
    img_size = (3, 224, 224)
    model = ro.ResNet(ro.BasicBlock, [2, 2, 2, 2])

    # print(model)
    summary(model, img_size, device="cpu")

    img = torch.randn(7, *img_size)
    z = model(img)
    print(z.shape)


def res_dec():
    print("-" * 40)
    model = rd.ResNet(rd.BasicBlock, [2, 2, 2, 2])

    # summary(model, (13, 1000), device="cpu")

    z = torch.randn(13, 1000)
    x = model(z)

    Color.print(x.shape, c=Color.red)


@singledispatch
def hoge(a):
    print("default")


@hoge.register
def _hoge(a: int, g: int):
    print("int")


@hoge.register
def _hoge(a: int, g: str):
    print("vdsa")


from functools import singledispatchmethod


class Negator:
    @singledispatchmethod
    @staticmethod
    def neg(arg, a):
        print("defautl")

    @neg.register
    def _(arg: int, a: str):
        print("A")

    @neg.register
    def _(arg: int, a: int):
        print("B")


# Negator.neg(3, 5)
# Negator.neg(6, "gaer")


from torch import nn


def haha():
    # target output size of 5x7
    m = nn.AdaptiveAvgPool2d((5, 7))
    input = torch.randn(1, 64, 8, 9)
    output = m(input)
    print(output.shape)
    # target output size of 7x7 (square)
    m = nn.AdaptiveAvgPool2d(7)
    input = torch.randn(1, 64, 10, 9)
    output = m(input)
    print(output.shape)
    # target output size of 10x7
    m = nn.AdaptiveAvgPool2d((None, 7))
    input = torch.randn(1, 64, 10, 9)
    print(nnio.get_size(m, tuple(input.shape)))
    output = m(input)
    print(output.shape)


def test11():
    enc = resnet2.ResNet18Enc(z_dim=dim_latent)
    img = torch.randn((7, 3, 256, 256))
    z = enc(img)
    print(z.shape)
    summary(enc, img.shape[-3:], device="cpu")
    print(f"{z.min().item():.4f}, {z.max().item():.4f}")

    dec = resnet2.ResNet18Dec(z_dim=dim_latent)
    # summary(dec, z.shape, device="cpu")
    x = dec(z)
    print(x.shape)
    print(f"{x.min().item():.4f}, {x.max().item():.4f}")


from source.models import from_stable_diffusion


def test12():
    from_stable_diffusion.Upsample.IOtest()

    # for h in range(100, 120):
    #     for w in range(100, 120):
    #         model = parts.Upsample(7, True)
    #         input = torch.randn(5, 7, h, w)
    #         output = model(input)

    #         print(f"I {tuple(input.shape)}, O {tuple(output.shape)}")
    #         In = (torch.tensor(input.shape) * 2)[-2:]
    #         Out = torch.tensor(output.shape)[-2:]
    #         assert (In == Out).all().item()


if __name__ == "__main__":

    # img64()
    # img256()
    # resnet18()
    # decoderV2()
    res_orig()
    # res_dec()
    # Foo.size(4)
    # hoge(4, 5)
    # hoge(4, "5")
    # a = np.floor(425.63)
    # print(type(a))
    # print(a)
    # haha()
    # test11()
    # test12()
