import common

common.set_path(__file__)

import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch import Tensor, nn, optim
from torchvision.io import read_image

import mypython.vision as mv
from models.mobile_unet import Masker, MobileUNet
from mypython.ai.train import train as mp_train
from mypython.ai.util import random_sample
from mypython.vision import convert_range
from seg_data import MaskingDataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # python train.py data_imgs/ee data_imgs/orange_complete.json
    train(sys.argv[1], sys.argv[2])


def for_show(imgs):
    imgs = convert_range(imgs, (0, 1), (0, 255)).detach_().cpu().numpy()
    return imgs


def train(imgs_dir, coco_file):
    batch_size = 5

    trainloader = MaskingDataLoader(
        imgs_dir,
        coco_file,
        batch_size=batch_size,
        device=device,
        cutout_and_random=None,
    )

    valid_imgs = []
    for i, fname in enumerate(Path(str(imgs_dir) + "_valid").glob("*")):
        valid_imgs.append(read_image(str(fname)))
        # if i == batch_size - 1:
        #     break
    valid_imgs = torch.stack(valid_imgs).to(device) / 255
    # print(valid_imgs.shape)
    # print(valid_imgs.min(), valid_imgs.max())

    model = Masker(out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), 2e-4)

    def post_epoch_fn(epoch, status):
        if epoch % 20 == 0:
            # trainloader.cutout_and_random = True
            img, mask = trainloader.sample_batch()

            input = random_sample(valid_imgs, batch_size)
            # input = img

            out = model.masked(input)
            mv.show_imgs_cv(
                for_show(torch.cat([input, out])),
                "masked",
                image_order="CHW",
                color_order="RGB",
                cols=len(input),
                block=False,
            )

    mp_train(
        model=model,
        optimizer=optimizer,
        trainloader=trainloader,
        epochs=1000,
        post_epoch_fn=post_epoch_fn,
        managed_dir="_unet_save/" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )

    # torch.save(model.state_dict(), "save/unet/unet_weight.pth")


if __name__ == "__main__":
    main()
