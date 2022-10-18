# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# Modified by Sierkinhane(sierkinhane@163.com)
# ------------------------------------------------------------------------------

import os
import random

import torch
import torch.utils.data as data
import pandas as pd
from PIL import Image
import numpy as np

from ..utils.transforms import fliplr_joints, crop, generate_target, transform_pixel
from imgaug import augmenters as iaa
import cv2
from scipy.ndimage import distance_transform_edt

BOUNDARIES = \
    {"WFLW": [
        [0, 32, 0],       # 轮廓     1
        [33, 37, 0],      # 右眉 上  2
        [38, 41, 1, 33],  # 右眉 下  3
        [42, 46, 0],      # 右眉 上  4
        [46, 50, 0],      # 右眉 下  5
        [51, 54, 0],      # 鼻梁     6
        [55, 59, 0],      # 鼻子下   7
        [76, 82, 0],      # 外嘴 上  8
        [82, 87, 1, 76],  # 外嘴 下  9
        [88, 92, 0],      # 内嘴 上  10
        [92, 95, 1, 88],  # 内嘴 下  11
        [60, 64, 0],      # 右眼 上  12
        [64, 67, 1, 60],  # 右眼 下  13
        [68, 72, 0],      # 左眼 上  14
        [72, 75, 1, 68],  # 左眼 下  15
    ]}
#
# BOUNDARIES_retified = \
#     {"WFLW": [
#         [0, 32, 0],       # 轮廓     1
#         [33, 37, 0],      # 右眉 上  2
#         [38, 41, 1, 33],  # 右眉 下  3
#         [42, 46, 0],      # 左眉 上  4
#         [46, 50, 0],      # 左眉 下  5
#         [51, 54, 0],      # 鼻梁     6
#         [55, 59, 0],      # 鼻子下   7
#         [60, 64, 0],      # 右眼 上  12
#         [64, 67, 1, 60],  # 右眼 下  13  插入 heatmap_96
#         [68, 72, 0],      # 左眼 上  14
#         [72, 75, 1, 68],  # 左眼 下  15  插入 heatmap_97
#         [76, 82, 0],      # 外嘴 上  8
#         [82, 87, 1, 76],  # 外嘴 下  9
#         [88, 92, 0],      # 内嘴 上  10
#         [92, 95, 1, 88],  # 内嘴 下  11
#         # + 两个眼中心的heatmap
#     ]}

class WFLW(data.Dataset):
    def __init__(self, cfg, is_train=True, transform=None):
        # specify annotation file for dataset
        if is_train:
            self.csv_file = cfg.DATASET.TRAINSET
        else:
            self.csv_file = cfg.DATASET.TESTSET

        self.is_train = is_train
        self.transform = transform
        self.data_root = cfg.DATASET.ROOT
        self.input_size = cfg.MODEL.IMAGE_SIZE
        self.output_size = cfg.MODEL.HEATMAP_SIZE
        self.sigma = cfg.MODEL.SIGMA
        self.bsigma = cfg.MODEL.SIGMA
        self.bmu = cfg.MODEL.BMU
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rot_factor = cfg.DATASET.ROT_FACTOR
        self.label_type = cfg.MODEL.TARGET_TYPE
        self.flip = cfg.DATASET.FLIP
        self.iaa_aug = cfg.DATASET.IAA_AUG
        self.num_boundaries = cfg.MODEL.NUM_BOUNDARIES

        # load annotations
        self.landmarks_frame = pd.read_csv(self.csv_file)

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        print("load {} images!".format(self.__len__()))

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):

        image_path = os.path.join(self.data_root,
                                  self.landmarks_frame.iloc[idx, 0])
        scale = self.landmarks_frame.iloc[idx, 1]

        center_w = self.landmarks_frame.iloc[idx, 2]
        center_h = self.landmarks_frame.iloc[idx, 3]
        center = torch.Tensor([center_w, center_h])

        pts = self.landmarks_frame.iloc[idx, 4:].values
        pts = pts.astype('float').reshape(-1, 2)

        scale *= 1.25
        nparts = pts.shape[0]
        img = np.array(Image.open(image_path).convert('RGB'), dtype=np.uint8)

        r = 0
        if self.is_train:
            if random.random() <= 0.5 and self.iaa_aug:
                aug = iaa.CoarseDropout(0.2, size_percent=0.02)
                img = aug(image=img)

            if random.random() <= 0.5 and self.iaa_aug:
                mul = np.random.uniform(0.5, 1.5)
                add = np.random.uniform(-25, 25)
                aug = iaa.MultiplyAndAddToBrightness(mul=mul, add=add)
                img = aug(image=img)

            img = img.astype(np.float32)

            scale = scale * (random.uniform(1 - self.scale_factor,
                                            1 + self.scale_factor))
            r = random.uniform(-self.rot_factor, self.rot_factor) \
                if random.random() <= 0.6 else 0
            if random.random() <= 0.5 and self.flip:
                img = np.fliplr(img)
                pts = fliplr_joints(pts, width=img.shape[1], dataset='WFLW')
                center[0] = img.shape[1] - center[0]
        else:
            img = img.astype(np.float32)

        img = crop(img, center, scale, self.input_size, rot=r)

        target = np.zeros((nparts, self.output_size[0], self.output_size[1]))
        tpts = pts.copy()

        # draw landmark heatmaps
        for i in range(nparts):
            if tpts[i, 1] > 0:
                tpts[i, 0:2] = transform_pixel(tpts[i, 0:2]+1, center,
                                               scale, self.output_size, rot=r)
                target[i] = generate_target(target[i], tpts[i]-1, self.sigma,
                                            label_type=self.label_type)

        # =================================================================================================#
        # BEGIN:                             generate boundary heatmaps                                    #
        # =================================================================================================#
        boundaries = np.ones((self.num_boundaries, self.output_size[0], self.output_size[1]))
        boundaries_t = np.ones((self.output_size[0], self.output_size[1]))  # store total boundary
        btpts = tpts.astype(np.int32)
        p = 0
        btpts -= 1
        for part in BOUNDARIES['WFLW']:
            for i in range(part[0], part[1]):
                if btpts[i, 1] > 0 and btpts[i, 0] > 0 and btpts[i + 1, 0] > 0 and btpts[i + 1, 1] > 0:
                    boundaries[p] = cv2.line(boundaries[p], tuple(btpts[i, :]), tuple(btpts[i + 1, :]),
                                             (0), 1)
                    boundaries_t = cv2.line(boundaries_t, tuple(btpts[i, :]), tuple(btpts[i + 1, :]), (0),
                                            1)
                    if i == (part[1] - 1) and part[2] == 1:
                        boundaries[p] = cv2.line(boundaries[p], tuple(btpts[part[1], :]),
                                                 tuple(btpts[part[3], :]), (0), 1)
                        boundaries_t = cv2.line(boundaries_t, tuple(btpts[part[1], :]),
                                                tuple(btpts[part[3], :]), (0), 1)
            p += 1
        boundaries[-1] = boundaries_t
        boundaries = boundaries.astype(np.float32)

        for i in range(boundaries.shape[0]):
            temp = boundaries[i]
            if len(temp[temp > 0]) != 0:
                dist_transform = distance_transform_edt(boundaries[i])
                boundaries[i] = np.exp(-(dist_transform - self.bmu) ** 2 / (2 * self.bsigma ** 2))
        # =================================================================================================#
        # END:                              generate boundary heatmaps                                     #
        # =================================================================================================#

        img = img.astype(np.float32)
        img = (img/255.0 - self.mean) / self.std
        img = img.transpose([2, 0, 1])
        target = torch.Tensor(target)
        tpts = torch.Tensor(tpts)
        center = torch.Tensor(center)
        boundaries = torch.Tensor(boundaries)

        meta = {'index': idx, 'center': center, 'scale': scale,
                'pts': torch.Tensor(pts), 'tpts': tpts}

        return img, target, boundaries, meta


if __name__ == '__main__':
    pass
