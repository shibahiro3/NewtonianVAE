import os
import sys

sys.path.append(os.pardir)

import sys
from copy import copy
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF

import mypython.ai.util as aiu
import mypython.vision as mv
from mypython import rdict
from mypython.pyutil import function_test
from simulation.env import ControlSuite
from tool import prepost


def main():
    test_with_env("point_mass-easy")
    test1()
    test_pre()
    # surprise()


device = torch.device("cuda")  # fast
# device = torch.device("cpu") # too slow

# batch_img = torch.randint(0, 255, (20, 1080, 1920, 3), dtype=torch.uint8, device=device) # わりと時間かかる
batch_img = torch.randint(0, 255, (20, 256, 256, 3), dtype=torch.uint8, device=device)
N = 50 * 300  # one loop * epoch
# N = 1000
orig_ = batch_img.flatten()[:3].detach().clone()


# pp = prepost.HandyForImage(bit_depth=5, size=(224, 224), out_range=[0, 1])
pp = prepost.HandyForImage()


def check_prepost(pre, post):
    assert type(pre) == torch.Tensor
    assert type(post) == np.ndarray
    rdict.show(dict(pre=pre, post=post))


@function_test
def test_with_env(env_):
    env = ControlSuite(env=env_, seed=1, max_episode_length=100, imgsize=[480, 480])
    obs = env.reset()
    rdict.show(obs)
    # env.render()

    img = next(iter(obs["camera"].values()))
    img_pre = pp.pre(img)
    img_post = pp.post(img_pre)
    check_prepost(img_pre, img_post)

    mv.show_imgs([img, img_post], show_size=True)
    plt.show()


@function_test
def test1():
    batch_img_pre = pp.pre(batch_img)
    batch_img_post = pp.post(batch_img_pre)
    check_prepost(batch_img_pre, batch_img_post)
    assert (batch_img.flatten()[:3] == orig_).all().item()


@function_test
def test_pre():
    for _ in range(N):
        batch_img_pre = pp.pre(batch_img)


@function_test
def surprise():
    x = torch.randint(0, 255, (5,), dtype=torch.uint8, device=device)
    x = np.random.randint(0, 255, (5,), dtype=np.uint8)
    # y1 = x * 2 / 255
    # y2 = x * 2.0 / 255
    y1 = x * 2
    y2 = x * 2.0
    print(x)
    print(y1)  # uint8 (<= 255)
    print(y2)  # OK (255 <)


if __name__ == "__main__":
    main()
