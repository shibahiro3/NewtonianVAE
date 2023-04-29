if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.pardir + os.sep)


from typing import Callable, Optional

import torch
import vit_pytorch
from torchsummary import summary

import models
from models import from_stable_diffusion, mobile_unet, parts, parts_backend
from mypython.ai.util import show_model_info
from mypython.pyutil import function_test, human_readable_byte
from mypython.terminal import Color

torch.set_grad_enabled(False)

# torch.set_default_dtype(torch.float32)
# torch.set_default_dtype(torch.float16)


def IOprinter(input: torch.Tensor, output: torch.Tensor):
    print(f"I {tuple(input.shape)}, O {tuple(output.shape)}")
    print(f"O min, max: {output.min().item():.4f}, {output.max().item():.4f}")


def IOcheck(
    input: torch.Tensor,
    output: torch.Tensor,
    in2out: Callable[[int], int],
):
    In = torch.tensor(input.shape)[-2:]
    Out = torch.tensor(output.shape)[-2:]

    Out_ = in2out(In)
    assert (Out == Out_).all().item()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def common(
    model: torch.nn.Module,
    in2out: Optional[Callable[[int], int]] = None,
    show=False,
    B=5,  # prime
    C=7,  # prime
    # HL=100,
    # HH=120,
    # WL=100,
    # WH=120,
    HL=64,
    HH=256,
    WL=67,  # prime
    WH=67,  # prime
    device=torch.device("cpu"),
):
    model.to(device)
    show_model_info(model, verbose=False)

    for h in range(HL, HH + 1):
        for w in range(WL, WH + 1):

            input = torch.randn(B, C, h, w).to(device)
            output = model(input)

            if show:
                IOprinter(input, output)

            if in2out is not None:
                IOcheck(input, output, in2out)


# ============================================================

# device = torch.device("cpu")
device = torch.device("cuda")


@function_test
def conv311(show=False):
    IC, OC = 7, 11  # prime
    model = from_stable_diffusion.conv311(IC, OC)
    common(model, show=show, C=IC)


@function_test
def conv320(show=False):
    IC, OC = 7, 11
    model = from_stable_diffusion.conv320(IC, OC)
    common(model, show=show, C=IC, in2out=lambda In: (In - 3) // 2 + 1)
    # in2out = lambda In: torch.floor((In - 3) / 2 + 1)


@function_test
def conv110(show=False):
    IC, OC = 7, 11
    model = from_stable_diffusion.conv110(IC, OC)
    common(model, show=show, C=IC)


@function_test
def upsample(show=False):
    IC, OC = 7, 11
    model = from_stable_diffusion.Upsample(IC, True)
    common(model, show=show, C=IC, in2out=lambda In: In * 2)


@function_test
def downsample(show=False):
    IC, OC = 7, 11
    model = from_stable_diffusion.Downsample(IC, with_conv=False)
    common(model, show=show, C=IC, in2out=lambda In: In // 2)


@function_test
def downsample_with_conv(show=False):
    IC, OC = 7, 11
    model = from_stable_diffusion.Downsample(IC, with_conv=False)
    common(model, show=show, C=IC, in2out=lambda In: In // 2)


@function_test
def simple_decoder(show=False):
    IC, OC = 64, 11
    model = from_stable_diffusion.SimpleDecoder(IC, OC)
    common(model, show=show, C=IC, in2out=lambda In: In * 2)


# @function_test
# def unet_double_conv(show=False):
#     IC, OC = 7, 11  # prime
#     model = parts.DoubleConv(IC, OC)
#     # common(model, show=show, C=IC, in2out=lambda x: x)
#     common(model, show=show, C=IC, HL=572, HH=572, in2out=lambda x: x)


@function_test
def vanilla_cnn64():
    dim_latent = 13
    B = 7
    img_size = 64

    img = torch.randn(B, 3, img_size, img_size).to(device)
    model = parts.ResNet(dim_latent, version="resnet18").to(device)

    summary(model, img.shape[-3:], device=device)
    show_model_info(model, verbose=False)

    z = model(img)

    IOprinter(img, z)

    assert z.shape == (B, dim_latent)


# @function_test
# def resnet_decoder():
#     dim_latent = 13
#     B = 7
#     img_size = 224
#     # img_size = 256

#     z = torch.randn(B, dim_latent).to(device)
#     model = parts.ResNetDecoder(dim_latent, img_size=img_size).to(device)

#     show_model_info(model, verbose=False)

#     x = model(z)

#     IOprinter(z, x)

#     assert x.shape == (B, 3, img_size, img_size)


@function_test
def resnet_official():
    dim_latent = 13
    B = 7
    img_size = 224
    # img_size = 256

    img = torch.randn(B, 3, img_size, img_size).to(device)
    model = parts.ResNet(dim_latent, version="resnet18").to(device)

    summary(model, img.shape[-3:], device=device)
    show_model_info(model, verbose=False)

    z = model(img)

    IOprinter(img, z)

    assert z.shape == (B, dim_latent)


@function_test
def vit():
    dim_latent = 13
    B = 7
    img_size = 224
    # img_size = 256

    img = torch.randn(B, 3, img_size, img_size).to(device)
    model = vit_pytorch.ViT(
        image_size=img_size,
        patch_size=32,
        num_classes=dim_latent,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1,
    ).to(device)

    summary(model, img.shape[-3:], device=device)
    show_model_info(model, verbose=False)

    z = model(img)

    IOprinter(img, z)

    assert z.shape == (B, dim_latent)


@function_test
def simple_vit():
    dim_latent = 13
    B = 7
    img_size = 224
    # img_size = 256

    img = torch.randn(B, 3, img_size, img_size).to(device)
    model = vit_pytorch.SimpleViT(
        image_size=img_size,
        patch_size=32,
        num_classes=dim_latent,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
    ).to(device)

    summary(model, img.shape[-3:], device=device)
    show_model_info(model, verbose=False)

    z = model(img)

    IOprinter(img, z)

    assert z.shape == (B, dim_latent)


@function_test
def encoder_v1():
    B = 7
    img_size = 64
    img = torch.randn(B, 3, img_size, img_size).to(device)
    enc = parts.EncoderV1(img_channels=3).to(device)
    out = enc(img)
    IOprinter(img, out)


@function_test
def decoder_v1():
    B = 7
    img = torch.randn(B, 1024, 8, 8).to(device)
    dec = parts.DecoderV1(img_channels=3).to(device)
    out = dec(img)
    IOprinter(img, out)


@function_test
def mobile_unet_():
    dim_latent = 13
    B = 7
    img_size = 64
    # img_size = 224

    img = torch.randn(B, 3, img_size, img_size).to(device)
    model = mobile_unet.MobileUNet(out_channels=1).to(device)

    out = model(img)

    summary(model, img.shape[-3:], device=device)
    IOprinter(img, out)


@function_test
def decoder_v2():
    dim_latent = 13
    B = 7
    z = torch.randn(B, dim_latent).to(device)
    model = parts.DecoderV2(dim_input=dim_latent).to(device)
    x = model(z)
    IOprinter(z, x)


@function_test
def irb_decoder_wrap():
    dim_latent = 13
    B = 7
    z = torch.randn(B, dim_latent).to(device)
    model = parts.IRBDecoderWrap(dim_input=dim_latent, img_size=224).to(device)
    x = model(z)
    IOprinter(z, x)


@function_test
def decoder_v3():
    dim_latent = 13
    B = 7
    z = torch.randn(B, dim_latent).to(device)
    model = parts.DecoderV3(dim_input=dim_latent, img_size=64).to(device)
    x = model(z)
    IOprinter(z, x)


@function_test
def decoder_v3():
    dim_latent = 13
    B = 7
    z = torch.randn(B, dim_latent).to(device)
    model = parts.DecoderV3(dim_input=dim_latent, img_size=64).to(device)
    x = model(z)
    IOprinter(z, x)


# @function_test
# def resnet_c():
#     ic = 3
#     img_size = 223
#     B = 7
#     x = torch.randn(B, ic, img_size, img_size).to(device)
#     model = parts.ResNetC().to(device)
#     z = model(x)
#     IOprinter(x, z)


@function_test
def upsample_conv_decoder_c():
    ic = 512
    B = 7
    z = torch.randn(B, ic, 2, 2).to(device)
    model = parts_backend.DecoderC_(in_c=ic, out_c=3).to(device)
    x = model(z)
    IOprinter(z, x)


@function_test
def autoencoder_v1():
    # img_size = 64
    img_size = 224
    # img_size = 256
    # img_size = 512
    # img_size = 768
    img = torch.randn(7, 3, img_size, img_size).to(device)
    enc = parts_backend.ResNetC().to(device)
    dec = parts.DecoderC(img_size=img_size).to(device)

    z = enc(img)
    out = dec(z)

    print("img:", img.shape)
    print("z:  ", z.shape)
    print("out:", out.shape)
    assert img.shape == out.shape

    print("⭐ enc")
    summary(enc, img.shape[-3:], device=device)
    print("⭐ dec")
    summary(dec, z.shape, device=device)


@function_test
def resnet_decoder():
    dim_latent = 63
    B = 7
    z = torch.randn(B, dim_latent).to(device)
    model = parts.ResnetDecoder(dim_input=dim_latent, img_size=224).to(device)
    x = model(z)
    IOprinter(z, x)


# @function_test
# def self_attention():
#     in_c = 8

#     for s in [64, 65, 66, 67, 68, 126, 127, 128, 129, 130, 131]:  # 224: torch.cuda.OutOfMemoryError
#         print("s:", s)
#         x = torch.randn(7, in_c, s, s).to(device)
#         model = parts_backend.SelfAttention(in_channels=in_c).to(device)
#         out, attention = model(x)
#         IOprinter(x, out)
#         print("atten:", attention.shape)


@function_test
def mobile_vit():
    # img_size = 64
    # img_size = 224
    # img_size = 128
    img_size = 256
    # img_size = 512
    # img_size = 768
    x = torch.randn(7, 3, img_size, img_size).to(device)
    model = parts.MobileViTWrap(11, img_size=img_size, param_version="xxs").to(device)
    print(count_parameters(model))
    out = model(x)
    IOprinter(x, out)


def mobile_unet_v2():
    in_c = 3

    # for s in [64, 65, 66, 67, 68, 126, 127, 128, 129, 130, 131]:
    for s in [64, 224, 256]:
        print("s:", s)
        x = torch.randn(7, in_c, s, s).to(device)
        model = mobile_unet.MobileUNet(in_channels=in_c, out_channels=in_c).to(device)
        out = model(x)
        print(count_parameters(model))
        IOprinter(x, out)


# @function_test
# def MaxUnpool2d_():
#     ic = 512
#     B = 7
#     z = torch.randn(B, ic, 2, 2).to(device)
#     model = torch.nn.MaxUnpool2d(3).to(device)
#     z1 = model(z)
#     IOprinter(z, z1)

if __name__ == "__main__":
    # unet_double_conv(True)

    # conv311()
    # conv320()
    # conv110()
    # upsample()
    # downsample()
    # downsample_with_conv()
    # simple_decoder()

    # vanilla_cnn64()

    # resnet_decoder()
    # resnet_official()

    # vit()
    # simple_vit()

    # mobile_unet_()

    # encoder_v1()
    # decoder_v1()
    # decoder_v2()
    # irb_decoder_wrap()
    # decoder_v3()

    # resnet_c()
    # upsample_conv_decoder_c()
    # autoencoder_v1()
    # self_attention()
    # mobile_vit()

    # mobile_unet_v2()
    resnet_decoder()
