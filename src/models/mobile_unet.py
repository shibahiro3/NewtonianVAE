"""
Reference:
    https://github.com/lafith/Mobile-UNet

    License: MIT
    https://github.com/lafith/Mobile-UNet/blob/main/LICENSE
"""


# Mobile UNet and Inverted Residual Block
# Author: Lafith Mattara
# Date: July 2022


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class InvertedResidualBlock(nn.Module):
    """
    inverted residual block used in MobileNetV2
    """

    def __init__(self, in_c, out_c, stride, expansion_factor=6, deconvolve=False):
        super(InvertedResidualBlock, self).__init__()
        # check stride value
        assert stride in [1, 2]
        self.stride = stride
        self.in_c = in_c
        self.out_c = out_c
        # Skip connection if stride is 1
        self.use_skip_connection = True if self.stride == 1 else False

        # expansion factor or t as mentioned in the paper
        ex_c = int(self.in_c * expansion_factor)
        if deconvolve:
            self.conv = nn.Sequential(
                # pointwise convolution
                nn.Conv2d(self.in_c, ex_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(ex_c),
                nn.ReLU6(inplace=True),
                # depthwise convolution
                nn.ConvTranspose2d(ex_c, ex_c, 4, self.stride, 1, groups=ex_c, bias=False),
                nn.BatchNorm2d(ex_c),
                nn.ReLU6(inplace=True),
                # pointwise convolution
                nn.Conv2d(ex_c, self.out_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(self.out_c),
            )
        else:
            self.conv = nn.Sequential(
                # pointwise convolution
                nn.Conv2d(self.in_c, ex_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(ex_c),
                nn.ReLU6(inplace=True),
                # depthwise convolution
                nn.Conv2d(ex_c, ex_c, 3, self.stride, 1, groups=ex_c, bias=False),
                nn.BatchNorm2d(ex_c),
                nn.ReLU6(inplace=True),
                # pointwise convolution
                nn.Conv2d(ex_c, self.out_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(self.out_c),
            )
        self.conv1x1 = nn.Conv2d(self.in_c, self.out_c, 1, 1, 0, bias=False)

    def forward(self, x):
        if self.use_skip_connection:
            # Skip connection is not used in decoder
            out = self.conv(x)
            if self.in_c != self.out_c:
                x = self.conv1x1(x)
            return x + out
        else:
            return self.conv(x)


def depthwise_conv(in_c, out_c, k=3, s=1, p=0):
    """
    optimized convolution by combining depthwise convolution and
    pointwise convolution.
    """
    conv = nn.Sequential(
        nn.Conv2d(in_c, in_c, kernel_size=k, padding=p, groups=in_c, stride=s),
        nn.BatchNorm2d(num_features=in_c),
        nn.ReLU6(inplace=True),
        nn.Conv2d(in_c, out_c, kernel_size=1),
    )
    return conv


def irb_bottleneck(in_c, out_c, n, s, t, d=False):
    """
    create a series of inverted residual blocks.
    """
    convs = []
    xx = InvertedResidualBlock(in_c, out_c, s, t, deconvolve=d)
    convs.append(xx)
    if n > 1:
        for i in range(1, n):
            xx = InvertedResidualBlock(out_c, out_c, 1, t, deconvolve=d)
            convs.append(xx)
    conv = nn.Sequential(*convs)
    return conv


def get_count(model):
    # simple function to get the count of parameters in a model.
    num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num


class MobileUNet(nn.Module):
    """
    Modified UNet with inverted residual block and depthwise seperable convolution
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()

        # encoding arm
        self.conv3x3 = depthwise_conv(in_channels, 32, p=1, s=2)
        self.irb_bottleneck1 = irb_bottleneck(32, 16, 1, 1, 1)
        self.irb_bottleneck2 = irb_bottleneck(16, 24, 2, 2, 6)
        self.irb_bottleneck3 = irb_bottleneck(24, 32, 3, 2, 6)
        self.irb_bottleneck4 = irb_bottleneck(32, 64, 4, 2, 6)
        self.irb_bottleneck5 = irb_bottleneck(64, 96, 3, 1, 6)
        self.irb_bottleneck6 = irb_bottleneck(96, 160, 3, 2, 6)
        self.irb_bottleneck7 = irb_bottleneck(160, 320, 1, 1, 6)
        self.conv1x1_encode = nn.Conv2d(320, 1280, kernel_size=1, stride=1)
        # decoding arm
        self.D_irb1 = irb_bottleneck(1280, 96, 1, 2, 6, True)
        self.D_irb2 = irb_bottleneck(96, 32, 1, 2, 6, True)
        self.D_irb3 = irb_bottleneck(32, 24, 1, 2, 6, True)
        self.D_irb4 = irb_bottleneck(24, 16, 1, 2, 6, True)
        self.DConv4x4 = nn.ConvTranspose2d(16, 16, 4, 2, 1, groups=16, bias=False)
        # Final layer: output channel number can be changed as per the usecase
        self.conv1x1_decode = nn.Conv2d(16, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        # Left arm/ Encoding arm
        # D1
        x1 = self.conv3x3(x)  # (32, 112, 112)
        x2 = self.irb_bottleneck1(x1)  # (16,112,112) s1
        x3 = self.irb_bottleneck2(x2)  # (24,56,56) s2
        x4 = self.irb_bottleneck3(x3)  # (32,28,28) s3
        x5 = self.irb_bottleneck4(x4)  # (64,14,14)
        x6 = self.irb_bottleneck5(x5)  # (96,14,14) s4
        x7 = self.irb_bottleneck6(x6)  # (160,7,7)
        x8 = self.irb_bottleneck7(x7)  # (320,7,7)
        x9 = self.conv1x1_encode(x8)  # (1280,7,7) s5

        # print("x6", x6.shape)
        # print("x4", x4.shape)
        # print("x3", x3.shape)
        # print("x2", x2.shape)
        # print("x9", x9.shape)

        # Right arm / Decoding arm with skip connections
        d1 = self.D_irb1(x9) + x6
        d2 = self.D_irb2(d1) + x4
        d3 = self.D_irb3(d2) + x3
        d4 = self.D_irb4(d3) + x2
        d5 = self.DConv4x4(d4)
        out = self.conv1x1_decode(d5)
        return out


class Masker(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()
        self.unet = MobileUNet(in_channels, out_channels)

    def forward(self, input, target):
        out = self.mask(input)
        L = F.mse_loss(out, target)
        # L = F.binary_cross_entropy(out, target)
        return L

    def mask(self, imgs):
        return torch.sigmoid(self.unet(imgs))

    def masked(self, imgs):
        return imgs * self.mask(imgs)


class MobileUNetWithLatent(MobileUNet):
    def __init__(self, img_size, ch=3):
        super().__init__(in_channels=ch, out_channels=ch)

        if type(img_size) == int:
            img_size = (img_size, img_size)

        self.ih = img_size[0]
        self.iw = img_size[1]

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        middle = int(self.zh * self.zw // 2)
        self.fc1 = nn.Linear(1, middle)
        self.fc2 = nn.Linear(middle, self.zh * self.zw)

    def forward(self, x):
        # Left arm/ Encoding arm
        # D1
        x1 = self.conv3x3(x)  # (32, 112, 112)
        x2 = self.irb_bottleneck1(x1)  # (16,112,112) s1
        x3 = self.irb_bottleneck2(x2)  # (24,56,56) s2
        x4 = self.irb_bottleneck3(x3)  # (32,28,28) s3
        x5 = self.irb_bottleneck4(x4)  # (64,14,14)
        x6 = self.irb_bottleneck5(x5)  # (96,14,14) s4
        x7 = self.irb_bottleneck6(x6)  # (160,7,7)
        x8 = self.irb_bottleneck7(x7)  # (320,7,7)
        x9 = self.conv1x1_encode(x8)  # (1280,7,7) s5

        print("x6", x6.shape)
        print("x4", x4.shape)
        print("x3", x3.shape)
        print("x2", x2.shape)
        print("x9", x9.shape)

        # Right arm / Decoding arm with skip connections
        d1 = self.D_irb1(x9) + x6
        d2 = self.D_irb2(d1) + x4
        d3 = self.D_irb3(d2) + x3
        d4 = self.D_irb4(d3) + x2
        d5 = self.DConv4x4(d4)
        out = self.conv1x1_decode(d5)
        return out

    def encode(self, x: torch.Tensor):
        pass

    def decode(self, z: torch.Tensor):
        pass


class IRBDecoder(nn.Module):
    """
    Input shape :  (B, in_c, zh, zw)
    Output shape : (B, out_c, 32*zh, 32*zw)

    Note:
        Skip connection is not used in decoder

    2 * 32 = 64
    7 * 32 = 224
    8 * 32 = 256
    16 * 32 = 512
    """

    def __init__(self, out_c=3, in_c=1280) -> None:
        super().__init__()
        self.D_irb1 = irb_bottleneck(in_c, 96, 1, 2, 6, True)
        self.D_irb2 = irb_bottleneck(96, 32, 1, 2, 6, True)
        self.D_irb3 = irb_bottleneck(32, 24, 1, 2, 6, True)
        self.D_irb4 = irb_bottleneck(24, 16, 1, 2, 6, True)
        self.DConv4x4 = nn.ConvTranspose2d(16, 16, 4, 2, 1, groups=16, bias=False)
        # Final layer: output channel number can be changed as per the usecase
        self.conv1x1_decode = nn.Conv2d(16, out_c, kernel_size=1, stride=1)

    def forward(self, z):
        d1 = self.D_irb1(z)
        d2 = self.D_irb2(d1)
        d3 = self.D_irb3(d2)
        d4 = self.D_irb4(d3)
        d5 = self.DConv4x4(d4)
        out = self.conv1x1_decode(d5)
        return out


if __name__ == "__main__":
    print("[MobileUNet]")

    # x = torch.rand((1, 3, 224, 224))
    # x = torch.rand((1, 3, 1024, 1024))
    # x = torch.rand((1, 3, 256, 256))
    x = torch.rand((1, 3, 64, 64))

    model = MobileUNet(out_channels=1)
    out = model(x)
    print(out.shape)

    z = torch.rand((1, 11, 5, 7))
    model = IRBDecoder(out_c=3, in_c=11)
    out = model(z)
    print(out.shape)
