import common

common.set_path(__file__)

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
from mypython.ai.util import BatchIndices
from mypython.terminal import Color


def main():
    # python seg_data.py data_imgs/ee data_imgs/orange_complete.json
    core(sys.argv[1], sys.argv[2])


def core(imgs_dir, coco_file):
    trainloader = MaskingDataLoader(
        imgs_dir,
        coco_file,
        batch_size=5,
        cutout_and_random=None,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    trainloader.sample_batch(verbose=False, show=True, debug=True)


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
        full_load: bool = True,
        cutout_and_random: Optional[bool] = False,  # None is mixed
        device=torch.device("cpu"),
    ):
        def has_duplicates(seq):
            return len(seq) != len(set(seq))

        self.images_dir = images_dir
        self.full_load = full_load
        self.cutout_and_random = cutout_and_random
        self.device = device

        self.coco = COCO(coco_file)
        self._image_ids = np.array(self.coco.getImgIds())
        cat_ids = self.coco.getCatIds()
        cat_ids = cat_ids[0]  # <<⭐ support only one category for cutout_and_random
        self._filenames = np.array(
            [ii["file_name"] for ii in self.coco.loadImgs(self._image_ids)]
        )  # correspond index
        super().__init__(0, len(self._image_ids), batch_size, shuffle)

        # print("ImgIds:", self.image_ids)
        # print("CatIds:", cat_ids)
        # print("finenames:", self._filenames)

        if full_load:
            self.imgs_tensor = self._load_imgs_from_fnames(self._filenames)

        not_supported = []
        for i, img_id in enumerate(self._image_ids):
            # mask_all = [None for _ in range(len(cat_ids))]
            anns_ids = self.coco.getAnnIds(imgIds=img_id, catIds=cat_ids)
            anns = self.coco.loadAnns(anns_ids)
            i_cat_ids = [ann["category_id"] for ann in anns]
            if has_duplicates(i_cat_ids):
                not_supported.append(i)

        if not_supported:
            Color.print("\nNot supported list", c=Color.red)
            print(f"number of images: {len(not_supported)}")
            print("id filename")
            for i in not_supported:
                print(self._image_ids[i], self._filenames[i])
            raise Exception(f"One category should have only one category_id")

    def _load_imgs(self, indices):
        if self.full_load:
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
            masks: N 1 H W   0 or 1
        """

        indices = super().__next__()
        imgs_orig = (self._load_imgs(indices) / 255).to(self.device)
        # imgs_orig = self._transform(imgs_orig)

        if self.cutout_and_random is None:
            cutout_and_random = bool(random.getrandbits(1))
        else:
            cutout_and_random = self.cutout_and_random

        if cutout_and_random:
            images = torch.rand_like(imgs_orig)
            # images = imgs_orig  # check round
            masks = torch.zeros((len(indices), 1, *imgs_orig.shape[-2:]), device=self.device)
        else:
            images = imgs_orig
            masks = torch.empty((len(indices), 1, *imgs_orig.shape[-2:]), device=self.device)
            images = self._transform(images)

        # TODO: parallel.. but training time is already few minitus.
        for i, img_id in enumerate(self._image_ids[indices]):

            anns_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(anns_ids)
            # for ann in anns:
            ann = anns[0]
            mask = self.coco.annToMask(ann)
            mask = torch.from_numpy(mask)

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

                mask = mask[BY : BY + BH, BX : BX + BW].to(self.device).to(torch.bool)  # HW
                mask = TF.resize(
                    mask.unsqueeze_(0), size=size, interpolation=TF.InterpolationMode.NEAREST
                ).squeeze_(0)

                imgs_o_ = imgs_orig[i, :, BY : BY + BH, BX : BX + BW]
                imgs_o_ = self._transform(imgs_o_)
                imgs_o_ = TF.resize(imgs_o_, size=size, interpolation=TF.InterpolationMode.NEAREST)

                # bg_ = images[i, :, BY : BY + BH, BX : BX + BW]
                # bg_ = TF.resize(bg_, size=size, interpolation=TF.InterpolationMode.NEAREST) # masked背景の引き伸ばしが起こる
                bg_ = torch.rand((3, *size), device=self.device)

                masked_img = imgs_o_ * mask + bg_ * ~mask  # CHW

                h, w = size
                x = np.random.randint(H - w + 1)
                y = np.random.randint(W - h + 1)
                images[i, :, y : y + h, x : x + w] = masked_img
                masks[i, 0, y : y + h, x : x + w] = mask  # bool to float32

            else:
                masks[i, 0] = mask

        return images, masks

    def sample_batch(
        self, batch_size: Union[int, str] = "same", verbose=False, show=False, debug=False
    ):
        batch_size_prev = self.batchsize
        if type(batch_size) == str:
            assert batch_size in ("same", "all")
            if batch_size == "all":
                self._B = self.datasize

        if debug:
            # for exception check
            for iter_ in range(20):
                for img_, mask_ in self:
                    pass

        img, mask = next(self)
        self.reset_indices()

        if verbose:
            print("img: ", img.shape, img.dtype, img.min(), img.max())
            print("mask:", mask.shape, mask.dtype, mask.min(), mask.max())

        if show:
            mv.show_imgs(
                mv.CHW2HWC(img).unbind() + mv.CHW2HWC(mask).unbind(),
                cols=self.batchsize,
                cmap="gray",
            )
            plt.show()

        self._B = batch_size_prev
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
    main()
