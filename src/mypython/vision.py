"""
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
from torch import Tensor

from mypython.ai.util import to_np

_NT = Union[np.ndarray, Tensor]

# Color order
def reverseRGB(imgs: _NT) -> _NT:
    return imgs[..., [2, 1, 0]]


# OpenCV: BGR (cv2.imread, cv2.imshow)
# matplotlib: RGB (plt.imshow)
# H, W order : same
RGB2BGR = reverseRGB
BGR2RGB = reverseRGB
cv2plt = BGR2RGB
plt2cv = RGB2BGR


def plt2cnn(imgs: _NT, in_size=None, out_size=None) -> Tensor:
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


def cv2cnn(imgs: np.ndarray) -> Tensor:
    return plt2cnn(cv2plt(imgs))


def cnn2plt(imgs: _NT) -> np.ndarray:
    """
    in  (N, RGB, H, W) (0 to 1)
    out (N, H, W, RGB) (0 to 255)
    """
    imgs = imgs * 255
    imgs = CHW2HWC(imgs)
    if type(imgs) == Tensor:
        imgs = imgs.detach().cpu().type(torch.uint8).numpy()
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
    return _transpose(x, axes)


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
    return _transpose(x, axes)


def _transpose(x: _NT, axes) -> _NT:
    if type(x) == Tensor:
        return x.permute(axes)
    if type(x) == np.ndarray:
        return x.transpose(axes)
    else:
        assert False


def clip(x, min, max):
    if type(x) == Tensor:
        return torch.clip(x, min=min, max=max)
    else:
        return np.clip(x, a_min=min, a_max=max)


def show_imgs(
    images: Union[np.ndarray, torch.Tensor, List],
    rows=None,
    cols=None,
    titles=None,
    fig=None,
    fontsize=7,
    show_size=False,
    lim=60,
    **kwargs,
):
    """
    左から右，1段降りて　の繰り返し
    画像サイズはバラバラでも問題ない

    Args:
        images:
            shape: [N, H, W, RGB], or [[H, W, RGB], ...] or [H, W, RGB] (1画像モード)
            dtypeがfloatなら 0 to 1, intなら 0 to 255 にすること．

    EXamples:
        images, labels = iter(trainloader).next()
        show_imgs(CHW2HWC(images), 8, 8, labels, cmap="gray")
    """

    if type(images) == np.ndarray or type(images) == torch.Tensor:
        # if len(images.shape) <= 3:  # 1画像モード
        if images.ndim <= 3:
            plt.imshow(to_np(images))
            plt.show()
            return

    if titles is not None:
        assert len(titles) == len(images)

    N = len(images)
    if N > lim:
        N = lim
        print(f"Limited to {lim} (origin len: ({len(images)}))")

    def rc(r_or_c):
        q, mod = divmod(N, r_or_c)
        if mod > 0:
            return q + 1
        else:
            return q

    if rows is None and cols is None:
        rows = int(np.sqrt(N))
        cols = rc(rows)
    elif rows is not None and cols is None:
        cols = rc(rows)
    elif rows is None and cols is not None:
        rows = rc(cols)

    assert N <= rows * cols

    if fig is None:
        fig = plt.figure()

    axes = fig.subplots(nrows=rows, ncols=cols)
    axes = axes.reshape(-1)
    for i in range(N):
        ax = axes[i]
        img = images[i]
        dtype = img.dtype
        # if dtype == torch.uint8 or dtype == np.uint8:
        # (N, H, W, RGB) row-major, float: 0 to 1  int: vmin=0 to vmax=255
        ax.imshow(to_np(img), **kwargs)
        # else:
        #     ax.imshow(images[i], **kwargs, vmin=0, vmax=1)

        if titles is not None:
            ax.set_title(str(titles[i]), fontsize=fontsize)

        if show_size:
            ax.set_ylabel(f"{img.shape[0]} px")  # H
            ax.set_xlabel(f"{img.shape[1]} px")  # W

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        # ax.set_axis_off()
        ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
        # ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)

    # 残りを消す
    for ax in axes[N:]:
        ax.set_axis_off()

    # plt.tight_layout()
    # plt.show()

    # plt.tight_layout()
    # plt.draw()
    # plt.pause(0.01)


def create_board(
    images: Union[np.ndarray, List[np.ndarray]],
    rows=None,
    cols=None,
    image_order="HWC",  # origin (images)
    color_order="BGR",  # origin (images)
    background_color=[255, 255, 255],  # RGB
    space=None,
    lim=60,
):
    """create one image
    画像サイズがバラバラでもOK
    inputが (H, W, 1) でもOK
    cv2.imshow : (H, W, BGR)

    Return:
        (H, W, BGR)
        input C=1 でも強制的に3チャンネルになる
        代入のブロードキャストが起きてるっぽい
    """

    def get_shape(img):
        if image_order == "CHW":
            c, h, w = img.shape[-3:]
        elif image_order == "HWC":
            h, w, c = img.shape[-3:]
        else:
            assert False

        return h, w, c

    def reshape(img):
        ret = img

        if type(ret) == Tensor:
            ret = ret.detach().cpu().numpy()

        if image_order == "HWC":  # default -> do nothing
            pass
        elif image_order == "CHW":
            ret = CHW2HWC(ret)
        else:
            assert False

        # now : HWC

        if color_order == "BGR":  # default -> do nothing
            pass
        elif color_order == "RGB":
            ret = ret[..., ::-1]
        else:
            assert False

        return ret

    def rc(r_or_c):
        q, mod = divmod(N, r_or_c)
        if mod > 0:
            return q + 1
        else:
            return q

    # ==========

    type_images = type(images)
    assert (
        type_images == list
        or type_images == tuple
        or type_images == np.ndarray
        or type_images == Tensor
    )

    N = len(images)
    if N > lim:
        N = lim
        images = images[:N]
        print(f"Limited to {lim} (origin len: ({len(images)}))")

    if rows is None and cols is None:
        rows = int(np.sqrt(N))
        cols = rc(rows)
    elif rows is not None and cols is None:
        cols = rc(rows)
    elif rows is None and cols is not None:
        rows = rc(cols)

    assert N <= rows * cols

    len(background_color) == 3

    hws = np.zeros((rows, cols, 2), dtype=int)
    for i in range(rows):
        for j in range(cols):
            idx = cols * i + j
            if idx == N:
                break
            assert images[idx].dtype == np.uint8
            h, w, c = get_shape(images[idx])
            assert h > 0 and w > 0
            hws[i, j] = [h, w]

    max_hs = hws[:, :, 0].max(1)  # colをmaxの対象rangeとする
    max_ws = hws[:, :, 1].max(0)  # rowをmaxの対象rangeとする

    if space is None:
        space = min(max_hs.max(), max_ws.max()) // 10
        if space < 0:
            space = 1

    # to BGR
    background_color = np.array(background_color[::-1], dtype=np.uint8)

    board = np.tile(
        background_color.reshape(1, 1, 3),
        (np.sum(max_hs) + space * (rows - 1), np.sum(max_ws) + space * (cols - 1), 1),
    )

    h_accum = 0
    for i in range(rows):
        w_accum = 0
        for j in range(cols):
            idx = cols * i + j
            if idx == N:
                break
            h, w, c = get_shape(images[idx])
            # print(f"{idx:3d}, {h:3d}, {w:3d}, {h_accum:3d}, {w_accum:3d}")
            board[h_accum : h_accum + h, w_accum : w_accum + w] = reshape(
                images[idx]
            )  # boradcast (C = 1 to 3)
            w_accum += max_ws[j] + space
        h_accum += max_hs[i] + space

    return board


def cv_wait(winname: str):
    while cv2.getWindowProperty(winname, cv2.WND_PROP_VISIBLE):
        cv2.waitKey(1)


def show_imgs_cv(
    images: Union[np.ndarray, List[np.ndarray]],
    winname: str,
    rows=None,
    cols=None,
    image_order="HWC",  # origin (images)
    color_order="BGR",  # origin (images)
    background_color=[255, 255, 255],  # RGB
    space=None,
    lim=60,
    block=True,
):
    """Faster"""

    board = create_board(
        images=images,
        rows=rows,
        cols=cols,
        image_order=image_order,
        color_order=color_order,
        background_color=background_color,
        space=space,
        lim=lim,
    )

    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.imshow(winname, board)

    if block:
        cv_wait(winname)
    else:
        cv2.waitKey(1)  # 小さすぎるとなぜか最初の反映が遅い
        # 先に
        # cv2.namedWindow("winname", cv2.WINDOW_NORMAL)
        # cv2.waitKey(1)
        # をやっておくと良い
        # tips: cv2.resizeWindow("winname", 1000, 700)


if __name__ == "__main__":
    import random

    def test1():
        # blue

        img = np.tile(
            np.array([255, 0, 0]).reshape((1, 1, 3)),
            (64, 64, 1),
        ).astype(np.uint8)

        print(img.shape)
        cv2.imshow("window", img)  # (H, W, BGR)
        cv_wait("window")

    def test_board():
        imgs = [
            np.tile(
                np.random.randint(0, 255, (1, 1, 3)),
                (random.randint(30, 60), random.randint(30, 60), 1),
            ).astype(np.uint8)
            for _ in range(23)
        ]
        print(imgs[0].shape)
        print(imgs[0].dtype)
        print(imgs[0][0, 0])

        show_imgs_cv(imgs, "winname", color_order="RGB", space=0)

    test1()
    test_board()
