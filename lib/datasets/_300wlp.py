from __future__ import print_function, absolute_import
import json
import os
from PIL import Image
import random
from lib.utils.transforms import *
from ..utils.transforms import fliplr_joints, crop, generate_target, transform_pixel, transform_pixel_3d
import imgaug.augmenters as iaa


class _300WLP(torch.utils.data.Dataset):
    def __init__(self, cfg, is_train=True):

        self.root = cfg.DATASET.ROOT['train'] if is_train else cfg.DATASET.ROOT['val']
        self.json_file = cfg.DATASET.JSON_FILE['train'] if is_train else cfg.DATASET.JSON_FILE['val']
        self.is_train = is_train
        self.input_size = cfg.MODEL.IMAGE_SIZE
        self.output_size = cfg.MODEL.HEATMAP_SIZE
        self.sigma = cfg.MODEL.SIGMA
        self.n_joints = cfg.MODEL.NUM_JOINTS
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rot_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.label_type = cfg.MODEL.TARGET_TYPE
        self.output_size = cfg.MODEL.HEATMAP_SIZE
        self.dataset_name = cfg.DATASET.DATASET
        self.iaa_aug = cfg.DATASET.IAA_AUG
        self.bsigma = cfg.MODEL.SIGMA
        self.bmu = cfg.MODEL.BMU
        self.num_boundaries = cfg.MODEL.NUM_BOUNDARIES

        self.mean = np.array(cfg.DATASET.MEAN, dtype=np.float32)
        self.std = np.array(cfg.DATASET.STD, dtype=np.float32)

        # create train/val split
        with open(self.json_file) as anno_file:
            self.anno = json.load(anno_file)

        self.data = []
        for idx, val in enumerate(self.anno):
            self.data.append(idx)

        print("load {} images!".format(self.__len__()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        a = self.anno[self.data[idx]]
        image_path = os.path.join(self.root, a['img_paths'])
        pts = np.array(a['landmarks']).astype('float')
        c = torch.Tensor(a['objpos'])
        s = a['scale_provided']
        bbox = a['bbox']
        nparts = pts.shape[0]
        if c[0] != -1:
            s *= 1.25

        img = np.array(Image.open(image_path).convert('RGB'), dtype=np.uint8)
        r=0
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

            s = s * (random.uniform(1 - self.scale_factor, 1 + self.scale_factor))
            r = random.uniform(-self.rot_factor, self.rot_factor) if random.random() <= 0.6 else 0

            # flip
            if random.random() <= 0.5 and self.flip:
                img = np.fliplr(img)
                pts = fliplr_joints(pts, width=img.shape[1], dataset=self.dataset_name)
                c[0] = img.shape[1] - c[0]
        else:
            img = img.astype(np.float32)

        # draw landmark heatmaps
        target = np.zeros((nparts, self.output_size[0], self.output_size[1]), dtype=np.float32)
        img = crop(img, c, s, self.input_size, rot=r)

        tpts = pts.copy()
        for i in range(self.n_joints):
            tpts[i, 0:3] = transform_pixel_3d(tpts[i, 0:3] + 1, c,
                                           s, self.output_size, self.output_size[0], rot=r)
            target[i] = generate_target(target[i], tpts[i, 0:2] - 1, self.sigma,
                                        label_type=self.label_type)

        img = img.astype(np.float32)
        img = (img/255. - self.mean) / self.std
        img = img.transpose([2, 0, 1])
        bbox = torch.Tensor(bbox)
        meta = {
            'index': idx, 'center': torch.Tensor(c), 'scale': s,
            'pts': torch.Tensor(pts),
            'tpts': tpts,
            'bbox':bbox
        }

        return img, target, 1, meta








