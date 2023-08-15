#!/usr/bin/env python3


import os
import pickle
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import common
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import rearrange
from torch import Tensor, nn, optim
from torchvision.io import read_image

import mypython.vision as mv
from models.mobile_unet import Masker, MobileUNet
from mypython.ai.train import train as mp_train
from mypython.ai.util import random_sample
from mypython.vision import convert_range
from unet_mask.seg_data import MaskingDataLoader


def sender(data: bytes, url):
    try:
        resp = requests.post(
            url=url,
            data=data,
            headers={"Content-Type": "application/octet-stream"},
        )
        # resp.raise_for_status()
    except:
        pass


def default_converter(x: Tensor, reverse_color=True):
    x = rearrange(x, "N C H W -> H (N W) C")
    x = convert_range(x, (0, 1), (0, 255))
    x = x.detach_().cpu().numpy().squeeze()  # for one channel
    if reverse_color:
        x = mv.reverseRGB(x)
    return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # ./src/train_unet.py data_imgs/ee/imgs data_imgs/ee/orange_complete.json data_imgs/ee_valid
    train(sys.argv[1], sys.argv[2], sys.argv[3])


def train(imgs_dir, coco_file, valid_dir):
    batch_size = 5

    cat_names = ["orange"]

    trainloader = MaskingDataLoader(
        imgs_dir,
        coco_file,
        batch_size=batch_size,
        device=device,
        cutout_and_random="mix",
        cat_names=cat_names,
        verbose=True,
    )
    trainloader.sample_batch(show=False, verbose=True)

    valid_imgs = []
    for i, fname in enumerate(Path(valid_dir).glob("*")):
        valid_imgs.append(read_image(str(fname)))
        # if i == batch_size - 1:
        #     break
    valid_imgs = torch.stack(valid_imgs).to(device) / 255

    model = Masker(out_channels=len(cat_names)).to(device)
    optimizer = optim.Adam(model.parameters(), 2e-4)

    def post_epoch_fn(epoch, status):
        if epoch % 20 == 0:
            # trainloader.cutout_and_random = True
            train_input, train_mask = trainloader.sample_batch()
            train_mask_out = model.mask(train_input)
            train_masked = train_input * train_mask_out

            valid_input = random_sample(valid_imgs, batch_size)
            valid_mask_out = model.mask(valid_input)
            valid_masked = valid_input * valid_mask_out

            sender(
                data=pickle.dumps(
                    {
                        "train input": default_converter(train_input),
                        "train mask": default_converter(train_mask, reverse_color=False),
                        "train mask out": default_converter(train_mask_out, reverse_color=False),
                        "train masked": default_converter(train_masked),
                        "valid input": default_converter(valid_input),
                        "valid mask out": default_converter(valid_mask_out, reverse_color=False),
                        "valid masked": default_converter(valid_masked),
                    }
                ),
                url="http://localhost:12345",
            )

    mp_train(
        model=model,
        optimizer=optimizer,
        trainloader=trainloader,
        epochs=2000,
        post_epoch_fn=post_epoch_fn,
        managed_dir="_unet_save/" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )

    # torch.save(model.state_dict(), "save/unet/unet_weight.pth")


if __name__ == "__main__":
    # t = ['c', 'a', 'a', 'b', 'c']
    # print(t.index('a'))
    main()
