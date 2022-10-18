import matplotlib.pyplot as plt
import cv2
import torch
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from torchvision.utils import make_grid

pairs = [[i, i + 1] for i in range(16)] + \
        [[i, i + 1] for i in range(17, 21)] + \
        [[i, i + 1] for i in range(22, 26)] + \
        [[i, i + 1] for i in range(36, 41)] + [[41, 36]] + \
        [[i, i + 1] for i in range(42, 47)] + [[47, 42]] + \
        [[i, i + 1] for i in range(27, 30)] + \
        [[i, i + 1] for i in range(31, 35)] + \
        [[i, i + 1] for i in range(48, 59)] + [[59, 48]] + \
        [[i, i + 1] for i in range(60, 67)] + [[67, 60]]

BOUNDARIES = \
    {'68':[
         [0,16,0],    # 脸颊
         [17,21,0],   # 右眉毛
         [22,26,0],   # 左眉毛
         [27,30,0],   # 鼻梁
         [31,35,0],   # 鼻子下
         [36,39,0],   # 右眼上
         [39,41,1,36],# 右眼 下
         [42,45,0],   # 左眼 上
         [45,47,1,42],# 左眼 下
         [48,54,0],   # 外嘴 上
         [54,59,1,48],# 外嘴 下
         [60,64,0],   # 内嘴 上
         [64,67,1,60],# 内嘴 下
    ],
    "98": [
        [0, 32, 0],  # 轮廓     1
        [33, 37, 0],  # 右眉 上  2
        [38, 41, 1, 33],  # 右眉 下  3
        [42, 46, 0],  # 右眉 上  4
        [46, 50, 0],  # 右眉 下  5
        [51, 54, 0],  # 鼻梁     6
        [55, 59, 0],  # 鼻子下   7
        [76, 82, 0],  # 外嘴 上  8
        [82, 87, 1, 76],  # 外嘴 下  9
        [88, 92, 0],  # 内嘴 上  10
        [92, 95, 1, 88],  # 内嘴 下  11
        [60, 64, 0],  # 右眼 上  12
        [64, 67, 1, 60],  # 右眼 下  13
        [68, 72, 0],  # 左眼 上  14
        [72, 75, 1, 68],  # 左眼 下  15
    ]
}

def show_joints(img, pts, show_idx=False, pairs=None):
    fig, ax = plt.subplots()
    ax.imshow(img)

    for i in range(pts.shape[0]):
        if pts[i, 2] > 0:
            ax.scatter(pts[i, 0], pts[i, 1], s=10, c='c', edgecolors='b', linewidth=0.3)
            if show_idx:
                plt.text(pts[i, 0], pts[i, 1], str(i))
            if pairs is not None:
                for p in pairs:
                    ax.plot(pts[p, 0], pts[p, 1], c='b', linewidth=0.3)

    plt.axis('off')
    plt.show()
    plt.close()


def show_joints_heatmap(img, target):
    img = cv2.resize(img, target.shape[1:])
    for i in range(target.shape[0]):
        t = target[i, :, :]
        plt.imshow(img, alpha=0.5)
        plt.imshow(t, alpha=0.5)
        plt.axis('off')
        plt.show()
    plt.close()


def show_joints_boundary(img, target):
    img = cv2.resize(img, target.shape[1:])
    for i in range(target.shape[0]):
        t = target[i, :, :]
        plt.imshow(img, alpha=0.5)
        plt.imshow(t, alpha=0.5)
        plt.axis('off')
        plt.show()
    plt.close()


# def show_joints_3d(img, pts, show_idx=False, pairs=None):
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#
#     ax.imshow(img)
#
#     for i in range(pts.size(0)):
#         if pts[i, 2] > 0:
#             ax.scatter(pts[i,0], pts[i,1], pts[i,2], s=5, c='c', edgecolors='b', linewidth=0.3)
#             if show_idx:
#                 plt.text(pts[i, 0], pts[i, 1], str(i))
#
#     plt.axis('off')
#     plt.show()
#     plt.close()
def show_joints_3d(predPts, pairs=None):
    ax = plt.subplot(111, projection='3d')

    view_angle = (-160, 30)
    if predPts.shape[1] > 2:
        ax.scatter(predPts[:, 2], predPts[:, 0], predPts[:, 1], s=5, c='c', marker='o', edgecolors='b', linewidths=0.5)
        # ax_pred.scatter(predPts[0, 2], predPts[0, 0], predPts[0, 1], s=10, c='g', marker='*')
        if pairs is not None:
            for p in pairs:
                ax.plot(predPts[p, 2], predPts[p, 0], predPts[p, 1], c='b', linewidth=0.5)
    else:
        ax.scatter([0] * predPts.shape[0], predPts[:, 0], predPts[:, 1], s=10, marker='*')
    ax.set_xlabel('z', fontsize=10)
    ax.set_ylabel('x', fontsize=10)
    ax.set_zlabel('y', fontsize=10)
    ax.view_init(*view_angle)
    plt.show()
    plt.close()


def save_plots(config, imgs, ppts_2d, ppts_3d, tpts_2d, tpts_3d, filename, nrows=4, ncols=4):
    # transform images
    mean = np.array(config.DATASET.MEAN, dtype=np.float32)
    std = np.array(config.DATASET.STD, dtype=np.float32)
    imgs = imgs.transpose(0, 2, 3, 1)
    imgs = (imgs * std + mean) * 255.
    imgs = imgs.astype(np.uint8)

    # plot 2d
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))

    cnt = 0
    for i in range(nrows):
        for j in range(ncols):
            # Output a grid of images
            axes[i, j].imshow(imgs[cnt])
            axes[i, j].scatter(ppts_2d[cnt, :, 0] * 4, ppts_2d[cnt, :, 1] * 4, s=10, c='c', edgecolors='w',
                               linewidth=1)
            axes[i, j].scatter(tpts_2d[cnt, :, 0] * 4, tpts_2d[cnt, :, 1] * 4, s=10, c='r', edgecolors='k',
                               linewidth=1)
            axes[i, j].axis('off')
            # if pairs is not None:
            #     for p in pairs:
            #         axes[i, j].plot(ppts_2d[cnt, p, 0] * 4, ppts_2d[cnt, p, 1] * 4, c='b', linewidth=0.5)
            #         axes[i, j].plot(tpts_2d[cnt, p, 0] * 4, tpts_2d[cnt, p, 1] * 4, c='r', linewidth=0.5)
            cnt += 1
    plt.savefig(filename + '_2d.png')
    plt.close()

    # plot 3d
    fig = plt.figure(figsize=(15, 15))
    for i in range(nrows * ncols):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection='3d')
        ax.scatter(ppts_3d[i, :, 2], ppts_3d[i, :, 0], ppts_3d[i, :, 1], s=10, color='b', edgecolor='w', alpha=0.6)
        ax.scatter(tpts_3d[i, :, 2], tpts_3d[i, :, 0], tpts_3d[i, :, 1], s=10, color='r', edgecolor='k', alpha=0.6)
        ax.view_init(elev=205, azim=110)
        # ax.axis('off')
        if pairs is not None:
            for p in pairs:
                ax.plot(ppts_3d[i, p, 2], ppts_3d[i, p, 0], ppts_3d[i, p, 1], c='b', linewidth=1)
                ax.plot(tpts_3d[i, p, 2], tpts_3d[i, p, 0], tpts_3d[i, p, 1], c='r', linewidth=1)
    plt.savefig(filename + '_3d.png')
    plt.close()


def save_plots_2d(config, imgs, ppts_2d, tpts_2d, filename, nrows=4, ncols=4):
    # transform images
    mean = np.array(config.DATASET.MEAN, dtype=np.float32)
    std = np.array(config.DATASET.STD, dtype=np.float32)
    imgs = imgs.transpose(0, 2, 3, 1)
    imgs = (imgs * std + mean) * 255.
    imgs = imgs.astype(np.uint8)

    # plot 2d
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))

    cnt = 0
    for i in range(nrows):
        for j in range(ncols):
            # Output a grid of images
            axes[i, j].imshow(imgs[cnt])
            axes[i, j].scatter(ppts_2d[cnt, :, 0] * 4, ppts_2d[cnt, :, 1] * 4, s=10, c='c', edgecolors='k', linewidth=1)
            axes[i, j].scatter(tpts_2d[cnt, :, 0] * 4, tpts_2d[cnt, :, 1] * 4, s=10, c='r', edgecolors='k', linewidth=1)
            axes[i, j].axis('off')
            if pairs is not None:
                for p in pairs:
                    axes[i, j].plot(ppts_2d[cnt, p, 0] * 4, ppts_2d[cnt, p, 1] * 4, c='b', linewidth=0.5)
                    axes[i, j].plot(tpts_2d[cnt, p, 0] * 4, tpts_2d[cnt, p, 1] * 4, c='r', linewidth=0.5)
            cnt += 1
    plt.savefig(filename + '_2d.png')
    plt.close()

def save_plots_validation(config, imgs, ppts_2d, tpts_2d, filename, fig=None, ax=None):
    # transform images
    mean = np.array(config.DATASET.MEAN, dtype=np.float32)
    std = np.array(config.DATASET.STD, dtype=np.float32)
    imgs = imgs.transpose(0, 2, 3, 1)
    imgs = (imgs * std + mean) * 255.
    imgs = imgs.astype(np.uint8)

    # plot 2d
    # print(ppts_2d)
    N = imgs.shape[0]
    ppts_2d -=1
    for i in range(N):
        fig, ax = plt.subplots()
        ax.imshow(imgs[i])
        if config.DATASET.DATASET == '300WLP':
            if pairs is not None:
                for p in pairs:
                    ax.plot(ppts_2d[i, p, 0] * 4, ppts_2d[i, p, 1] * 4, c='c', linewidth=2)
        ax.scatter(ppts_2d[i, :, 0] * 4, ppts_2d[i, :, 1] * 4, s=20, c='c', edgecolors='k', linewidth=2)
        # ax.scatter(tpts_2d[i, :, 0] * 4, tpts_2d[i, :, 1] * 4, s=15, c='r', edgecolors='k', linewidth=1)
        ax.axis('off')

        #         ax.plot(tpts_2d[i, p, 0] * 4, tpts_2d[i, p, 1] * 4, c='r', linewidth=0.5)
        # print(filename + '{}_2d.png'.format(i))
        plt.savefig(filename + '{}_2d.png'.format(i))
        plt.close()

def gausian(x, miu=1, sigma=0.2):

    return torch.exp(-(x-miu)**2 / (2*sigma**2))
def activation(x, sigma=0.2):
    miu = 1
    l_mask = torch.where(x <= miu, torch.full_like(x, 1), torch.full_like(x, 0))
    # otherwise
    o_mask = torch.where(x > miu, torch.full_like(x, 1), torch.full_like(x, 0))

    return l_mask * gausian(x, miu=miu, sigma=sigma) + o_mask * 1
from matplotlib.backends.backend_pdf import PdfPages

def save_boundary_validation(config, imgs, boundary, gt_boundary, ppts_2d, tpts_2d, filename, fig=None, ax=None):
    # transform images
    mean = np.array(config.DATASET.MEAN, dtype=np.float32)
    std = np.array(config.DATASET.STD, dtype=np.float32)
    imgs = imgs.transpose(0, 2, 3, 1)
    imgs = (imgs * std + mean) * 255.
    imgs = imgs.astype(np.uint8)
    ppts_2d -= 1
    tpts_2d -= 1
    # plot 2d

    # print(ppts_2d)
    N = imgs.shape[0]
    for i in range(N):
        cv2.imwrite(filename +'{}_img.png'.format(i), imgs[i][...,::-1])
        pdf = PdfPages(filename + '{}_2d.pdf'.format(i))
        fig, ax = plt.subplots()
        b = cv2.resize(boundary[i][-1], (256,256))
        # print(b.shape)
        b = activation(torch.from_numpy(b)).numpy()
        ax.imshow(imgs[i], alpha=0.7)
        ax.imshow(b, alpha=0.7)
        ax.scatter(ppts_2d[i, :, 0]*4, ppts_2d[i, :, 1]*4, s=20, c='c', edgecolors='k', linewidth=2)
        # ax.scatter(tpts_2d[i, :, 0] * 4, tpts_2d[i, :, 1] * 4, s=15, c='r', edgecolors='k', linewidth=1)
        ax.axis('off')
        # if pairs is not None:
        #     for p in pairs:
        #         ax.plot(ppts_2d[i, p, 0] * 4, ppts_2d[i, p, 1] * 4, c='b', linewidth=0.5)
        #         ax.plot(tpts_2d[i, p, 0] * 4, tpts_2d[i, p, 1] * 4, c='r', linewidth=0.5)
        print(filename + '{}_2d.png'.format(i))
        plt.savefig(filename + '{}_2d.png'.format(i))
        pdf.savefig()
        plt.close()
        pdf.close()

def norm_(feature_map, lim=[0,1]):

    for i in range(feature_map.shape[0]):
        f = feature_map[i]
        max_ = np.max(f)
        min_ = np.min(f)
        feature_map[i] = (lim[1] - lim[0])/(max_-min_)*(f-min_)

    return feature_map

def save_feature_map(feature_map, filename):
    feature_map = feature_map[0]
    feature_map = norm_(feature_map)
    N = feature_map.shape[0]
    for i in range(N):
        pdf = PdfPages(filename + '{}_features.pdf'.format(i))
        fig, ax = plt.subplots()
        ax.xaxis.set_ticks_position('top')  # 设置x轴刻度到上方
        im = ax.imshow(feature_map[i])
        plt.axis('off')
        plt.colorbar(im)
        pdf.savefig()
        pdf.close()
        plt.savefig(filename + '{}_features.png'.format(i))
        plt.close()

def save_plots_3d(config, ppts_3d, tpts_3d, filename, nrows=4, ncols=4):
    # plot 3d
    fig = plt.figure(figsize=(15, 15))
    for i in range(nrows * ncols):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection='3d')
        ax.scatter(ppts_3d[i, :, 2], ppts_3d[i, :, 0], ppts_3d[i, :, 1], s=10, color='b', edgecolor='w', alpha=0.6)
        # ax.scatter(tpts_3d[i, :, 2], tpts_3d[i, :, 0], tpts_3d[i, :, 1], s=10, color='r', edgecolor='k', alpha=0.6)
        ax.view_init(elev=205, azim=110)
        # ax.axis('off')
        if pairs is not None:
            for p in pairs:
                ax.plot(ppts_3d[i, p, 2], ppts_3d[i, p, 0], ppts_3d[i, p, 1], c='b', linewidth=1)
                # ax.plot(tpts_3d[i, p, 2], tpts_3d[i, p, 0], tpts_3d[i, p, 1], c='r', linewidth=1)
    plt.savefig(filename + '_3d.png')
    plt.close()


def save_joints_boundary(config, imgs, target, tpts_2d, filename, nrows=4, ncols=4):
    # transform images
    mean = np.array(config.DATASET.MEAN, dtype=np.float32)
    std = np.array(config.DATASET.STD, dtype=np.float32)
    imgs = imgs.transpose(0, 2, 3, 1)
    imgs = (imgs * std + mean) * 255.
    imgs = imgs.astype(np.uint8)

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

def demo_plot(cfg, img, box, points, color=None, rec=False, line=False):

    if color is None:
        colors = (list(np.random.choice(range(256), size=3)))
        color = [int(colors[0]), int(colors[1]), int(colors[2])]

    # choose comfortable radius and thickness
    bh, bw = box[3]-box[1], box[2]-box[0]
    barea = bh*bw
    h, w, _ = img.shape
    iarea = w*h

    ratio = iarea/barea
    r = int(ratio*(4/3.94))
    # print(iarea/barea)
    # r = int(sarea/160)
    thickness = int(iarea/120)

    if thickness == 0:
        thickness = 1
    if r == 0:
        r = 1

    if rec:
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (123,104,238), thickness=2)
    for i in range(points.shape[0]):
        center = points[i]
        cv2.circle(img, (center[0], center[1]), 4, (123,104,238), thickness=-1)
        cv2.circle(img, (center[0], center[1]), 4, (0,0,0), thickness=2)
    if line:
        for part in BOUNDARIES[str(cfg.MODEL.NUM_JOINTS)]:
            for i in range(part[0], part[1]):
                if points[i, 1] > 0 and points[i, 0] > 0 and points[i + 1, 0] > 0 and points[i + 1, 1] > 0:
                    cv2.line(img, tuple(points[i, :2]), tuple(points[i + 1, :2]),
                                             color, 2)
                    if i == (part[1] - 1) and part[2] == 1:
                        cv2.line(img, tuple(points[part[1], :2]),
                                                 tuple(points[part[3], :2]), color, thickness=2)
    return img.copy()
