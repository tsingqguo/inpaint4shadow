import json
import os
import random

import numpy as np
import scipy
import torch
import torchvision.transforms.functional as F
from PIL import Image
from scipy.misc import imread
from skimage.color import gray2rgb
from torch.utils.data import DataLoader


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, gt_path, mask_path, mask2_path, shadow_path, augment=True, training=True):
        super(Dataset, self).__init__()
        self.augment = augment
        self.training = training
        self.gt_list = self.load_flist(gt_path)
        self.mask_list = self.load_flist(mask_path)
        self.mask2_list = self.load_flist(mask2_path)
        self.shadow_list = self.load_flist(shadow_path)

        self.input_size = config.INPUT_SIZE
        self.sigma = config.SIGMA
        self.mask = config.MASK
        self.nms = config.NMSMASK_REVERSE

        self.reverse_mask = config.MASK_REVERSE
        self.mask_threshold = config.MASK_THRESHOLD

        print('training:{}  gt:{}  shadow:{}  mask:{}'.format(training, gt_path, shadow_path, mask_path))

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, index):
        item = self.load_item(index)
        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):
        size = self.input_size

        img = imread(self.gt_list[index])
        shadow = imread(self.shadow_list[index])
        mask = imread(self.mask_list[index])
        mask2 = imread(self.mask2_list[index])

        # gray to rgb
        if len(img.shape) < 3:
            img = gray2rgb(img)
            shadow = gray2rgb(shadow)

        # resize/crop if needed
        if self.augment:
            imgh, imgw = img.shape[0:2]
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = random.randint(0, imgw - side)

            img = self.resize_random(img, size, size, j, i, side)
            shadow = self.resize_random(shadow, size, size, j, i, side)
            mask = self.resize_random(mask, size, size, j, i, side)
        else:
            img = self.resize(img, size, size, centerCrop=False)
            shadow = self.resize(shadow, size, size, centerCrop=False)
            mask = self.resize(mask, size, size, centerCrop=False)
            mask2 = self.resize(mask2, size, size, centerCrop=False)

        mask = (mask > 255 * 0.9).astype(np.uint8) * 255
        mask2 = (mask2 > 255 * 0.9).astype(np.uint8) * 255

        # augment data
        if self.augment and np.random.binomial(1, 0.5) > 0:
            img = img[:, ::-1, ...]
            mask = mask[:, ::-1, ...]
            shadow = shadow[:, ::-1, ...]

        name = self.gt_list[index].split('/')[-1]
        self.name = name

        return self.to_tensor(img), self.to_tensor(mask), self.to_tensor(mask2),self.to_tensor(shadow)


    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def resize(self, img, height, width, centerCrop=True):
        imgh, imgw = img.shape[0:2]

        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        img = scipy.misc.imresize(img, [height, width], 'cubic')

        return img

    def resize_random(self, img, height, width, j, i, side):
        img = img[j:j + side, i:i + side, ...]

        img = scipy.misc.imresize(img, [height, width], 'cubic')

        return img


    def load_flist(self, flist):
        if flist is None:
            return []
        with open(flist, 'r') as j:
            f_list = json.load(j)
            return f_list


    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item