#!/usr/bin/env python3


import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms

import mypython.vision as mv
from mypython import rdict
from mypython.ai.util import SequenceDataLoader
from tool import paramsmanager


def main():
    # fmt: off
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("--n-imgs", type=int, default=20)
    parser.add_argument("--save-dir", type=str) # if not specified, only show. no save
    args = parser.parse_args()
    # fmt: on

    core(**vars(args))


def core(config, n_imgs: int, save_dir):
    """
    ランダムに選ばれたエピソードデータ群からランダムな時刻で選ばれた画像をn_imgs枚表示、保存する
    """

    params = paramsmanager.Params(config)
    loader = SequenceDataLoader(
        patterns=params.train.path,
        # batch_size=100,
        keypaths=[
            ["camera", "ee1"],
            ["camera", "ee2"],
        ],
    )
    batch = loader.sample_batch(verbose=True)
    imgs = torch.stack(list(batch["camera"].values()))
    imgs = imgs.flatten(end_dim=-4)
    indices = random.sample(range(imgs.shape[0]), n_imgs)
    imgs = imgs[indices]

    # to uint8, HWBGR (for opencv saving)
    imgs = imgs.detach().cpu().numpy()

    print(imgs.shape)
    mv.show_imgs_cv(imgs, "Sample images")

    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(imgs):
            cv2.imwrite(str(Path(save_dir, f"{i}.png")), img)


if __name__ == "__main__":
    main()
