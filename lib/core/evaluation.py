# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# Modified by Sierkinhane(sierkinhane@163.com)
# ------------------------------------------------------------------------------

import math
import torch
import numpy as np

from ..utils.transforms import transform_preds

def get_preds(scores):
    """
    get predictions from score maps in torch Tensor
    return type: torch.LongTensor
    """
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx = idx.view(scores.size(0), scores.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0] - 1) % scores.size(3) + 1
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / scores.size(3)) + 1

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask
    return preds


def compute_nme(cfg, preds, meta):

    targets = meta['pts']
    if cfg.DATASET.DATASET == '300WLP':
        targets = targets[:, :, :2]
    preds = preds.numpy()
    target = targets.cpu().numpy()

    N = preds.shape[0]
    L = preds.shape[1]
    rmse = np.zeros(N)

    for i in range(N):
        pts_pred, pts_gt = preds[i, ], target[i, ]
        if L == 19:  # aflw
            d = meta['box_size'][i]
        elif L == 29:  # cofw
            # d = np.linalg.norm(pts_gt[8, ] - pts_gt[9, ])
            d = np.linalg.norm(pts_gt[16, ] - pts_gt[17, ]) # this for calculating inter-pupil
        elif L == 68:  # 300w
            if cfg.DATASET.DATASET == 'COFW68' or cfg.DATASET.DATASET == '300WLP':
                d = np.sqrt(meta['bbox'][i][2] * meta['bbox'][i][3])
            else:
                # interoccular
                d = np.linalg.norm(pts_gt[36, ] - pts_gt[45, ])
                # interpupil
                # rp = (pts_gt[36,] + pts_gt[37,] + pts_gt[38,] + pts_gt[39,] + pts_gt[40,] + pts_gt[41,]) / 6
                # lp = (pts_gt[42,] + pts_gt[43,] + pts_gt[44,] + pts_gt[45,] + pts_gt[46,] + pts_gt[47,]) / 6
                # d = np.linalg.norm(rp - lp)

        elif L == 98:
            d = np.linalg.norm(pts_gt[60, ] - pts_gt[72, ])
        else:
            raise ValueError('Number of landmarks is wrong')
        rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1)) / (d * L)

    return rmse


def decode_preds(config, output, center, scale, res):
    coords = get_preds(output)  # float type
    # print(coords.shape)
    coords = coords.cpu()
    # pose-processing
    for n in range(coords.size(0)):
        for p in range(coords.size(1)):
            hm = output[n][p]
            px = int(math.floor(coords[n][p][0]))
            py = int(math.floor(coords[n][p][1]))
            if (px > 1) and (px < res[0]) and (py > 1) and (py < res[1]):
                diff = torch.Tensor([hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1]-hm[py - 2][px - 1]])
                coords[n][p] += diff.sign() * .25
    coords += 0.5
    preds = coords.clone()

    # Transform back
    for i in range(coords.size(0)):
        preds[i] = transform_preds(config, coords[i], center[i], scale[i], res)

    if preds.dim() < 3:
        preds = preds.view(1, preds.size())

    return preds

def decode_preds_vis(config, output, center, scale, res):
    coords = get_preds(output)  # float type
    coords = coords.cpu()
    # pose-processing
    for n in range(coords.size(0)):
        for p in range(coords.size(1)):
            hm = output[n][p]
            px = int(math.floor(coords[n][p][0]))
            py = int(math.floor(coords[n][p][1]))
            if (px > 1) and (px < res[0]) and (py > 1) and (py < res[1]):
                diff = torch.Tensor([hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1]-hm[py - 2][px - 1]])
                coords[n][p] += diff.sign() * .25

    vis = coords.clone()
    coords += 0.5
    preds = coords.clone()
    # Transform back
    for i in range(coords.size(0)):
        preds[i] = transform_preds(config, coords[i], center[i], scale[i], res)

    if preds.dim() < 3:
        preds = preds.view(1, preds.size())

    return preds, vis+0.5

# def compute_niou(x, y, d, threshold=[0, 0.25, 0.5, 0.75, 1], factor=[0.1, 0.2, 0.3, 0.4]):
#     """
#     compute normalized Interception over Union for evaluating boundary estimaton
#     x: groundtruth boundary heatmaps. (dtype: np.array)
#     y: predicted boundary heatmaps. (dtype: np.array)
#     """
#     x = sigmoid(x)
#
#     N = x.shape[0]
#     B = x.shape[1]
#     iou = np.zeros(N)
#     for i in range(N):
#         for k in range(B):
#             for j in range(len(threshold)-1):
#
#                 a, b = threshold[j], threshold[j+1]
#                 x_rows, x_cols = list(np.where((x[i, k]>a) & (x[i, k]<=b)))
#                 xidx = set(zip(list(x_rows), list(x_cols)))
#
#                 y_rows, y_cols = list(np.where((y[i, k]>a) & (y[i, k]<=b)))
#                 yidx = set(zip(list(y_rows), list(y_cols)))
#
#                 inter = len(list(xidx.intersection(yidx)))
#                 union = len(list(xidx.union(yidx)))
#                 # iou[i] += factor[j] * inter / union
#                 iou[i] += inter / union
#         iou[i] = iou[i] / (len(threshold)-1) / B
#
#     niou = iou / np.exp(d)
#
#     return np.mean(niou)

def compute_niou(x, y, d, threshold=[0, 0.25, 0.5, 0.75, 1], factor=[0.1, 0.2, 0.3, 0.4]):
    """
    compute normalized Interception over Union for evaluating boundary estimaton
    x: groundtruth boundary heatmaps. (dtype: np.array)
    y: predicted boundary heatmaps. (dtype: np.array)
    """
    # x = activation(x)

    N = x.shape[0]
    B = x.shape[1]
    iou = np.zeros(N)
    for i in range(N):
        for k in range(B):
            for j in range(len(threshold)-1):

                a, b = threshold[j], threshold[j+1]
                x_rows, x_cols = list(np.where((x[i, k]>a) & (x[i, k]<=b)))
                xidx = set(zip(list(x_rows), list(x_cols)))

                y_rows, y_cols = list(np.where((y[i, k]>a) & (y[i, k]<=b)))
                yidx = set(zip(list(y_rows), list(y_cols)))

                inter = len(list(xidx.intersection(yidx)))
                union = len(list(xidx.union(yidx)))
                iou[i] += factor[j] * inter / union

        iou[i] /= B

    niou = iou / np.exp(d)

    return np.mean(niou)

