"""
import mypython.img_util as imu

cv2.imread : (*, H, W, BGR) (0 to 255)
cnn : (*, RGB, H, W) (0 to 1) (floatにしないとcnnは受け入れない)
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as VF

_NT = Union[np.ndarray, torch.Tensor]


def cv2plt(imgs):
    """
    in  (*, H, W, BGR)
    out (*, H, W, RGB)
    """
    return BGR2RGB(imgs)


def plt2cnn(imgs: _NT, in_size=None, out_size=None) -> torch.Tensor:
    """
    in  (N, H, W, RGB) (0 to 255)
    out (N, RGB, H, W) (0 to 1)
    """
    if in_size is not None:
        assert imgs.shape[1] >= in_size

    if type(imgs) == np.ndarray:
        imgs = torch.from_numpy(imgs)  # imgs.copy() が要る場合がある

    imgs = HWC2CHW(imgs)
    if in_size is not None:
        imgs = VF.center_crop(imgs, (in_size, in_size))
    if out_size is not None:
        imgs = VF.resize(imgs, (out_size, out_size))
    imgs = imgs / 255.0
    return imgs.float()


def cv2cnn(imgs: np.ndarray) -> torch.Tensor:
    return plt2cnn(cv2plt(imgs))


def cnn2plt(imgs: _NT) -> np.ndarray:
    """
    in  (N, RGB, H, W) (0 to 1)
    out (N, H, W, RGB) (0 to 255)
    """
    imgs = imgs * 255
    imgs = CHW2HWC(imgs)
    if type(imgs) == torch.Tensor:
        imgs = imgs.cpu().type(torch.uint8).numpy()
    elif type(imgs) == np.ndarray:
        imgs = imgs.astype(np.uint8)
    else:
        assert False

    return imgs


def cnn2cv(imgs: _NT) -> np.ndarray:
    """
    in  (N, RGB, H, W) (0 to 1)
    out (N, H, W, BGR) (0 to 255)
    """
    imgs = imgs * 255
    imgs = CHW2HWC(imgs)
    imgs = BGR2RGB(imgs)
    imgs = imgs.cpu().type(torch.uint8).numpy()
    return imgs


def CHW2HWC(x: _NT) -> _NT:
    """
    Args:
        x: shape: [..., C, H, W]
    """

    assert x.ndim >= 3

    i = x.ndim - 3
    axes = tuple(range(0, x.ndim - 3)) + (1 + i, 2 + i, 0 + i)
    return _CHW_HWC_axis(x, axes)


def HWC2CHW(x: _NT) -> _NT:
    """
    Args:
        x: shape: [..., H, W, C] (OpenCV read)

    Returns:
        arr shape: [..., C, H, W] (Conv2D, torchvision.transforms.functional input)
    """

    assert x.ndim >= 3

    i = x.ndim - 3
    axes = tuple(range(0, x.ndim - 3)) + (2 + i, 0 + i, 1 + i)
    return _CHW_HWC_axis(x, axes)


def _CHW_HWC_axis(x: _NT, axes) -> _NT:
    if type(x) == np.ndarray:
        return x.transpose(axes)
    if type(x) == torch.Tensor:
        return x.permute(axes)
    else:
        assert False


def BGR2RGB(imgs: _NT) -> _NT:
    return imgs[..., [2, 1, 0]]


def RGB2BGR(imgs: _NT) -> _NT:
    return BGR2RGB(imgs)
