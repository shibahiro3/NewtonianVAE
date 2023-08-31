from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch import Tensor

import mypython.ai.util as aiu
import mypython.vision as mv
from mypython.terminal import Color


class PrePostBase:
    def pre(self, x) -> Tensor:
        """For NN input"""
        return x

    def post(self, x) -> np.ndarray:
        """For visualization"""
        return x


class HandyForImage(PrePostBase):
    def __init__(
        self,
        *,
        size: Optional[List[int]] = None,
        out_range: list = [-0.5, 0.5],
        bit_depth: Optional[int] = None,
        src_channel_order="HWC",  # or "CHW"
        src_reverse_color_order=False,  # BGR2RGB (RGB2BGR) to RGB2BGR (BGR2RGB)
        out_reverse_color_order=False,
        suppress_warning=False,
    ) -> None:
        """
        GPUなしだと論外レベルに遅い
        """

        super().__init__()
        self.size = tuple(size) if size is not None else None
        self.bit_depth = bit_depth
        self.out_range = tuple(out_range)
        self.to_4dim = To4Dim()
        self.suppress_warning = suppress_warning
        self.src_channel_order = src_channel_order
        self.src_reverse_color_order = src_reverse_color_order
        self.out_reverse_color_order = out_reverse_color_order

    def pre(self, img: Union[np.ndarray, torch.Tensor]) -> Tensor:
        """
        For NN input

        Input: Data
            shape=(*, src_channel_order) range=[0, 255]  (uint8)

        Output: For CNN input
            shape=(*, C, H, W)           range=out_range (torch.float32)

        uint8にfloat32はinplaceできない
        """
        if self.src_channel_order == "HWC":
            img = mv.HWC2CHW(img)

        if not self.suppress_warning:
            C = img.shape[-3]
            if C >= 32:  # 1 (monochrome) or 3 (RGB) or 4 (RGBA) or some concat
                Color.print(
                    f"WARNING: [preprocess] Is the image input size correct? Input channel: {C}",
                    c=Color.code.coral,
                )
        if not (img.dtype == np.uint8 or img.dtype == torch.uint8):
            raise TypeError(f"Input type is not uint8: ({img.dtype})")

        img = aiu.to_torch(img)

        if (self.size is not None) and (img.shape[-2:] != self.size):
            img = self.to_4dim.pre(img)
            img = TF.resize(img, self.size, interpolation=TF.InterpolationMode.NEAREST)
            img = self.to_4dim.post(img)

        if self.src_reverse_color_order:
            img = img[..., [2, 1, 0], :, :]

        if self.bit_depth is not None:
            # Quantise to given bit depth and centre
            img = img.div(2 ** (8 - self.bit_depth)).floor_().div_(2**self.bit_depth).sub_(0.5)
            # Dequantise (to approx. match likelihood of PDF of continuous images vs. PMF of discrete images)
            img.add_(torch.rand_like(img).div_(2**self.bit_depth))  # not noise meaning ..?

            if self.out_range != (-0.5, 0.5):
                img = mv.convert_range(img, (-0.5, 0.5), self.out_range)
        else:
            img = mv.convert_range(img, (0, 255), self.out_range)

        return img

    def post(self, img) -> np.ndarray:
        """
        For visualization

        Input:  NN output
            shape=(*, C, H, W) range=out_range

        Output: Image for opencv
            shape=(*, H, W, C) range=[0, 255] (numpy uint8)
        """
        if not self.suppress_warning:
            C = img.shape[-3]
            if C >= 32:  # 1 (monochrome) or 3 (RGB) or 4 (RGBA) or some concat
                Color.print(
                    f"WARNING: [postprocess] Is the image input size correct? Input channel: {C}",
                    c=Color.code.coral,
                )

        img = aiu.to_numpy(img)
        img = mv.CHW2HWC(img)
        # img = TF.resize(img)

        if self.bit_depth is not None:
            if self.out_range != (-0.5, 0.5):
                img = mv.convert_range(img, self.out_range, (-0.5, 0.5))
            img = np.clip(
                np.floor((img + 0.5) * 2**self.bit_depth) * 2 ** (8 - self.bit_depth), 0, 255
            ).astype(np.uint8)

        else:
            img = mv.convert_range(img, self.out_range, (0, 255))

        if self.out_reverse_color_order:
            img = img[..., [2, 1, 0]]

        return img


class To4Dim:
    """(*, a, b, c) -> (n, a, b, c) -> (*, a, b, c)
    For CNN input, transforms
    """

    def __init__(self) -> None:
        self._left_dims = ()

    def pre(self, x: Tensor):
        if x.ndim > 4:
            self._left_dims = x.shape[:-3]
            return x.flatten(end_dim=-4)
        return x

    def post(self, x: Tensor):
        if len(self._left_dims) > 0:
            return x.reshape(self._left_dims + x.shape[-3:])
        return x


def add_gaussian_noise_(x: Tensor, mean, std):
    # https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745
    return x.add_(torch.randn_like(x).mul_(std).add_(mean))
