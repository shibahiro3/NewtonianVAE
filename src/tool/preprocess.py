"""
学習時の高カロリーなpreprocessは時間の無駄
"""

import torchvision.transforms.functional as TF
from torch import Tensor

from mypython import rdict


def preprocess1(batch_data):
    # rdict.show(batch_data["camera"], "cameras")
    rdict.apply_(batch_data["camera"], lambda x: _resize(x, (64, 64)))
    rdict.apply_(batch_data["camera"], lambda x: x / 255.0 - 0.5)
    return batch_data


def resize64(batch_data):
    # データが512のものだとクソ遅くなる　同じ64なら速いまま
    # データは事前に変換しておくべき
    # データを前もって変換しておくのは、要は1エポック分を事前に変換しておくことと等しい
    # これを都度適用するのは時間の無駄
    rdict.apply_(batch_data["camera"], lambda x: _resize(x, (64, 64)))
    return batch_data


def one_tenth_action(batch_data):
    batch_data["action"] /= 10.0
    return batch_data


def _resize(x: Tensor, size):
    OH, OW = size
    T, B, C, H, W = x.shape
    x = TF.resize(x.reshape(-1, C, H, W), size)
    x = x.reshape(T, B, C, OH, OW)
    return x
