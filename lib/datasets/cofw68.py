# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# Modified by Sierkinhane(sierkinhane@163.com)
# ------------------------------------------------------------------------------

import math
import torch
import torch.utils.data as data
import numpy as np

from hdf5storage import loadmat
from ..utils.transforms import fliplr_joints, crop, generate_target, transform_pixel

class COFW68(data.Dataset):

    def __init__(self, cfg, is_train=True, transform=None):
        # specify annotation file for dataset
        if is_train:
            self.mat_file = cfg.DATASET.TRAINSET
        else:
            self.mat_file = cfg.DATASET.TESTSET

        self.is_train = is_train
        self.transform = transform
        self.data_root = cfg.DATASET.ROOT
        self.input_size = cfg.MODEL.IMAGE_SIZE
        self.output_size = cfg.MODEL.HEATMAP_SIZE
        # load annotations
        self.mat = loadmat(self.mat_file)
        if is_train:
            self.images = self.mat['IsTr']
        else:
            self.images = self.mat['IsT']

        self.bboxes = loadmat(cfg.DATASET.BBOXES)['bboxes']
        self.pts = np.zeros((507, 68, 2), dtype=np.float64)
        for i in range(507):
            point = loadmat(cfg.DATASET.POINTS + '{}_points.mat'.format(i+1))['Points']
            self.pts[i] = point

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        print("load {} images!".format(self.__len__()))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img = self.images[idx][0]

        if len(img.shape) == 2:
            img = img.reshape(img.shape[0], img.shape[1], 1)
            img = np.repeat(img, 3, axis=2)

        pts = self.pts[idx]
        bbox = self.bboxes[idx]
        xmin = np.min(pts[:, 0])
        xmax = np.max(pts[:, 0])
        ymin = np.min(pts[:, 1])
        ymax = np.max(pts[:, 1])

        center_w = (math.floor(xmin) + math.ceil(xmax)) / 2.0
        center_h = (math.floor(ymin) + math.ceil(ymax)) / 2.0

        scale = max(math.ceil(xmax) - math.floor(xmin), math.ceil(ymax) - math.floor(ymin)) / 200.0
        center = torch.Tensor([center_w, center_h])

        scale *= 1.25
        nparts = pts.shape[0]

        r = 0
        img = img.astype(np.float32)

        img = crop(img, center, scale, self.input_size, rot=r)

        target = np.zeros((nparts, self.output_size[0], self.output_size[1]))
        tpts = pts.copy()

        # for i in range(nparts):
        #     if tpts[i, 1] > 0:
        #         tpts[i, 0:2] = transform_pixel(tpts[i, 0:2]+1, center,
        #                                        scale, self.output_size, rot=r)
        #         target[i] = generate_target(target[i], tpts[i]-1, self.sigma,
        #                                     label_type=self.label_type)

        img = img.astype(np.float32)
        img = (img/255 - self.mean) / self.std
        img = img.transpose([2, 0, 1])
        target = torch.Tensor(target)
        tpts = torch.Tensor(tpts)
        center = torch.Tensor(center)
        bbox = torch.Tensor(bbox)

        meta = {'index': idx, 'center': center, 'scale': scale,
                'pts': torch.Tensor(pts), 'tpts': tpts, 'bbox': bbox}

        return img, target, 1, meta


if __name__ == '__main__':

    pass
