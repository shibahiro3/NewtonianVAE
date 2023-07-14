# import common

# common.set_path(__file__)

import random
import sys
from pathlib import Path
from typing import Optional, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# import torchvision
# torchvision.datasets.ImageFolder
import torchvision
import torchvision.transforms.functional as TF
from pycocotools.coco import COCO
from torchvision.io import read_image

import mypython.vision as mv
from models.mobile_unet import Masker
from mypython import rdict
from mypython.ai.util import BatchIndices
from mypython.terminal import Color


class MaskingDataLoader(BatchIndices):
    """
    torchvision.datasets.CocoDetection
    torchvision.datasets.CocoCaptions
    """

    def __init__(
        self,
        images_dir: str,
        coco_file: str,
        batch_size: int,
        shuffle: bool = True,
        load_all: bool = False,
        cutout_and_random: str = "none",
        device=torch.device("cpu"),
    ):
        assert cutout_and_random in ("none", "only", "mix")

        self.images_dir = images_dir
        self.load_all = load_all
        self.cutout_and_random = cutout_and_random
        self.device = device

        self.coco = COCO(coco_file)
        self._image_ids = np.array(self.coco.getImgIds())  # use slice
        self._cat_ids = self.coco.getCatIds()
        self._filenames = np.array([ii["file_name"] for ii in self.coco.loadImgs(self._image_ids)])

        print("ImgIds:", self._image_ids)
        print("CatIds:", self._cat_ids)
        print("finenames:", self._filenames)

        super().__init__(0, len(self._filenames), batch_size, shuffle)

        if load_all:
            self.imgs_tensor = self._load_imgs_from_fnames(self._filenames)

    def _load_imgs(self, indices):
        if self.load_all:
            return self.imgs_tensor[indices]
        else:
            return self._load_imgs_from_fnames(self._filenames[indices])

    def _load_imgs_from_fnames(self, filenames):
        imgs_tensor = []
        for fname in filenames:
            imgs_tensor.append(read_image(str(Path(self.images_dir, fname))))
        return torch.stack(imgs_tensor)  # NCHW, torch.uint8

    @staticmethod
    def _transform(imgs):
        imgs = TF.adjust_brightness(imgs, np.random.uniform(0.8, 1.2))
        # print(imgs.min(), imgs.max())
        return imgs

    def __next__(self):
        """
        Returns
            images:  N C H W   [0, 1]
            masks: N CO H W   0 or 1, CO = len(cat_ids)
        """

        indices = super().__next__()
        imgs_orig = (self._load_imgs(indices) / 255).to(self.device)
        # imgs_orig = self._transform(imgs_orig)

        if self.cutout_and_random == "mix":
            cutout_and_random = bool(random.getrandbits(1))
        elif self.cutout_and_random == "none":
            cutout_and_random = False
        elif self.cutout_and_random == "only":
            cutout_and_random = True

        if cutout_and_random:
            images = torch.rand_like(imgs_orig)
            # images = imgs_orig  # check round
            masks = torch.zeros(
                (len(indices), len(self._cat_ids), *imgs_orig.shape[-2:]),
                dtype=torch.bool,
                device=self.device,
            )
        else:
            images = imgs_orig
            masks = torch.empty(
                (len(indices), len(self._cat_ids), *imgs_orig.shape[-2:]),
                dtype=torch.bool,
                device=self.device,
            )
            images = self._transform(images)

        # TODO: parallel.. but training time is already few minitus.
        for ii, img_id in enumerate(self._image_ids[indices]):

            for ci, cat_id in enumerate(self._cat_ids):
                anns_ids = self.coco.getAnnIds(imgIds=img_id, catIds=cat_id)
                anns = self.coco.loadAnns(anns_ids)

                for ai, ann in enumerate(anns):
                    mask = self.coco.annToMask(ann)
                    mask = torch.from_numpy(mask).to(torch.bool).to(self.device)

                    if cutout_and_random:  # いらん処理でしたぁ
                        H, W = imgs_orig.shape[-2:]
                        BX, BY, BW, BH = ann["bbox"]
                        ratio = np.random.uniform(0.5, 1.5)
                        size = (np.array([BH, BW]) * ratio).astype(int)
                        if H < size[0]:
                            size[0] = H
                        if W < size[1]:
                            size[1] = W
                        size = tuple(size)

                        mask = mask[BY : BY + BH, BX : BX + BW]  # HW
                        mask = TF.resize(
                            mask.unsqueeze_(0),
                            size=size,
                            interpolation=TF.InterpolationMode.NEAREST,
                        ).squeeze_(0)

                        imgs_o_ = imgs_orig[ii, :, BY : BY + BH, BX : BX + BW]
                        imgs_o_ = self._transform(imgs_o_)
                        imgs_o_ = TF.resize(
                            imgs_o_, size=size, interpolation=TF.InterpolationMode.NEAREST
                        )

                        # bg_ = images[i, :, BY : BY + BH, BX : BX + BW]
                        # bg_ = TF.resize(bg_, size=size, interpolation=TF.InterpolationMode.NEAREST) # masked背景の引き伸ばしが起こる
                        bg_ = torch.rand((3, *size), device=self.device)

                        masked_img = imgs_o_ * mask + bg_ * ~mask  # CHW

                        h, w = size
                        x = np.random.randint(H - w + 1)
                        y = np.random.randint(W - h + 1)
                        images[ii, :, y : y + h, x : x + w] = masked_img

                        if ai == 0:
                            masks[ii, ci, y : y + h, x : x + w] = mask  # bool to float32
                        else:
                            masks[ii, ci, y : y + h, x : x + w] |= mask

                    else:
                        if ai == 0:
                            masks[ii, ci] = mask
                        else:
                            masks[ii, ci] |= mask

        masks = masks.to(images.dtype)
        return images, masks

    def sample_batch(
        self, batch_size: Union[int, str] = "same", verbose=False, show=False, debug=False
    ):
        batch_size_prev = self.batchsize
        if type(batch_size) == str:
            assert batch_size in ("same", "all")
            if batch_size == "all":
                self.set_batchsize(self.datasize)

        if debug:
            # for exception check
            for iter_ in range(20):
                for img_, mask_ in self:
                    pass

        img, mask = next(self)
        self.reset_indices()

        if verbose:
            rdict.show(dict(img=img, mask=mask), "batch data")

        if show:
            m = []
            for c in range(mask.shape[-3]):
                for n in range(mask.shape[0]):
                    m.append(mask[n, c])

            # imgs
            # categorical id 1 mask
            # categorical id 2 mask
            # ...
            mv.show_imgs(
                list(mv.CHW2HWC(img).unbind()) + m,
                cols=self.batchsize,
                cmap="gray",
            )
            plt.show()

        self.set_batchsize(batch_size_prev)
        return img, mask


def mask_unet(pth):
    masker = Masker(out_channels=1)
    masker.load_state_dict(torch.load(pth))
    print("unet loaded")

    def preprocess_(batchdata):
        with torch.no_grad():
            for cam_name in batchdata["camera"].keys():
                imgs = batchdata["camera"][cam_name]
                T, B, C, H, W = imgs.shape
                imgs = masker.masked(imgs.reshape(-1, C, H, W))
                imgs = imgs.reshape(T, B, C, H, W)
                batchdata["camera"][cam_name] = imgs
        return batchdata

    return masker, preprocess_


if __name__ == "__main__":
    # python seg_data.py data_imgs/ee data_imgs/orange_complete.json

    imgs_dir = sys.argv[1]
    coco_file = sys.argv[2]

    trainloader = MaskingDataLoader(
        imgs_dir,
        coco_file,
        batch_size=5,
        cutout_and_random="none",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    trainloader.sample_batch(verbose=True, show=True, debug=False)
