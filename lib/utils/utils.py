# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn), Tianheng Cheng(tianhengcheng@gmail.com)
# Modified by Sierkinhane(sierkinhane@163.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path

import torch
import torch.optim as optim
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simps

PAIRS ={'300W':
        [[i, i + 1] for i in range(16)] + \
        [[i, i + 1] for i in range(17, 21)] + \
        [[i, i + 1] for i in range(22, 26)] + \
        [[i, i + 1] for i in range(36, 41)] + [[41, 36]] + \
        [[i, i + 1] for i in range(42, 47)] + [[47, 42]] + \
        [[i, i + 1] for i in range(27, 30)] + \
        [[i, i + 1] for i in range(31, 35)] + \
        [[i, i + 1] for i in range(48, 59)] + [[59, 48]] + \
        [[i, i + 1] for i in range(60, 67)] + [[67, 60]]
        ,
        'WFLW':
        [[i, i + 1] for i in range(32)] + \
        [[i, i + 1] for i in range(33, 37)] + \
        [[i, i + 1] for i in range(38, 41)] + [[37, 38]] +\
        [[i, i + 1] for i in range(42, 46)] + \
        [[i, i + 1] for i in range(46, 50)] + [[42, 50]] + \
        [[i, i + 1] for i in range(51, 54)] + \
        [[i, i + 1] for i in range(55, 59)] + \
        [[i, i + 1] for i in range(76, 82)] + \
        [[i, i + 1] for i in range(82, 87)] + [[87, 76]] + \
        [[i, i + 1] for i in range(88, 92)] + \
        [[i, i + 1] for i in range(92, 95)] + [[88, 95]] +\
        [[i, i + 1] for i in range(60, 64)] +\
        [[i, i + 1] for i in range(64, 67)] + [[60, 67]] +\
        [[i, i + 1] for i in range(68, 72)] +\
        [[i, i + 1] for i in range(72, 75)] + [[68, 75]]
        }

Boundary = {"WFLW":[
         [0,32,0],    # 轮廓     1
         [33,37,0],   # 右眉 上  2
         [38,41,1,33],# 右眉 下  3
         [42,46,0],   # 右眉 上  4
         [46,50,0],   # 右眉 下  5
         [51,54,0],   # 鼻梁     6
         [55,59,0],   # 鼻子下   7
         [76,82,0],   # 外嘴 上  8
         [82,87,1,76],# 外嘴 下  9
         [88,92,0],   # 内嘴 上  10
         [92,95,1,88],# 内嘴 下  11
         [60,64,0],   # 右眼 上  12
         [64,67,1,60], # 右眼 下  13
         [68,72,0],   # 左眼 上  14
         [72,75,1,68], # 左眼 下  15
         ]}

def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
                        (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR
        )
    elif cfg.TRAIN.OPTIMIZER == 'rmsprop':
        optimizer = optim.RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            alpha=cfg.TRAIN.RMSPROP_ALPHA,
            centered=cfg.TRAIN.RMSPROP_CENTERED
        )

    return optimizer

def save_joints_boundary(config, imgs, target, tpts_2d, filename, nrows=4, ncols=4):
    # transform images
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    imgs = imgs.transpose(0, 2, 3, 1)
    imgs = (imgs * std + mean) * 255.
    imgs = imgs.astype(np.uint8)
    pairs = PAIRS[config.DATASET.DATASET]
    # plot 2d
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))
    cnt = 0
    for i in range(nrows):
        for j in range(ncols):
            # Output a grid of images
            b = target[cnt, -1, :, :]
            b = cv2.resize(b, (256,256))
            axes[i, j].imshow(imgs[cnt], alpha=0.8)
            axes[i, j].imshow(b, alpha=0.5)
            axes[i, j].scatter(tpts_2d[cnt, :, 0] * 4, tpts_2d[cnt, :, 1] * 4, s=10, c='r', edgecolors='k', linewidth=1)
            axes[i, j].axis('off')

            if pairs is not None:
                for p in pairs:
                    # axes[i, j].plot(ppts_2d[cnt, p, 0] * 4, ppts_2d[cnt, p, 1] * 4, c='b', linewidth=0.5)
                    axes[i, j].plot(tpts_2d[cnt, p, 0] * 4, tpts_2d[cnt, p, 1] * 4, c='r', linewidth=0.5)
            cnt += 1
    plt.savefig(filename + '_boundary.png')
    plt.close()

def AUCError(errors, failureThreshold=0.1, step=0.0001, plot_curve=True):

    nErrors = len(errors)
    xAxis = list(np.arange(0., failureThreshold + step, step))
    ced = [float(np.count_nonzero([errors <= x])) / nErrors for x in xAxis]
    AUC = simps(ced, x=xAxis) / failureThreshold
    failureRate = 1. - ced[-1]
    print("AUC @ {0}: {1}".format(failureThreshold, AUC))
    print("Failure rate: {0}".format(failureRate))
    if plot_curve:
        plt.plot(xAxis, ced)
        plt.show()

def sacle_boxes(boxes):
    """
    bboxes: (x,4) ((x1,y1,x2,y2)) np.array
    """
    center = (boxes[:, 0:2] + boxes[:, 2:4]) / 2
    w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    w_, h_ = (w / 2), (h / 2)
    # x1, y1
    boxes[:, 0], boxes[:, 1] = center[:, 0] - w_*1.5, center[:, 1] - h_ * 0.8,
    # x2, y2
    boxes[:, 2], boxes[:, 3] = center[:, 0] + w_*1.5, center[:, 1] + h_ * 1.2,

    return boxes

def sacle_boxes(boxes):
    """
    bboxes: (x,4) ((x1,y1,x2,y2)) np.array
    """
    center = (boxes[:, 0:2] + boxes[:, 2:4]) / 2
    w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    w_, h_ = (w / 2), (h / 2)
    # x1, y1
    boxes[:, 0], boxes[:, 1] = center[:, 0] - w_*1.5, center[:, 1] - h_ * 0.8,
    # x2, y2
    boxes[:, 2], boxes[:, 3] = center[:, 0] + w_*1.5, center[:, 1] + h_ * 1.2,

    return boxes

class Smoother(object):
    def __init__(self, v0=0, beta=0.95, threshold=10):

        self.v0 = v0
        self.beta = beta
        self.num_face = 0
        self.flag = 0
        self.mean_loc = 0
        self.threshold = threshold

    def smooth(self, points):

        N = points.shape[0]

        ## 初始化
        if self.flag == 0:
            self.flag = 1
            self.mean_loc = np.mean(points)
            self.num_face = N
            self.v0 = np.zeros(points.shape)
            self.beta = np.ones(points.shape) * self.beta

        ## 追踪时，人脸数量变多时重新追踪
        elif self.flag == 1 and N != self.num_face:
            self.mean_loc = np.mean(points)
            self.num_face = N
            self.v0 = np.zeros(points.shape)
            self.beta = np.ones(points.shape) * self.beta

        ## 人脸位置突然变化时重新追踪
        elif np.linalg.norm(self.mean_loc - np.mean(points)) >= self.threshold:
            self.mean_loc = np.mean(points)
            self.num_face = N
            self.v0 = np.zeros(points.shape)
            self.beta = np.ones(points.shape) * self.beta

        self.v0 = self.beta * self.v0 + (1-self.beta) * points

        return self.v0



