# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# Modified by Sierkinhane(sierkinhane@163.com)
# ------------------------------------------------------------------------------

import os
import pprint
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lib.datasets import get_dataset
from lib.core import function
from lib.utils import utils
from lib.models.tab import get_face_alignment_net
from shutil import copyfile
from easydict import EasyDict as edict
import yaml
import numpy as np


def parse_args():

    parser = argparse.ArgumentParser(description='Train TAB')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        required=True, type=str)
    parser.add_argument('--stem', default='vgg', help='experiments', type=str)
    parser.add_argument('--fusion', default='2x', help='experiments', type=str)
    parser.add_argument('--fs_type', default='multi-scale', help='experiments', type=str)
    parser.add_argument('--resume', default=False, help='resume the checkpoint', required=False, type=bool)
    parser.add_argument('--seed', default=4329, help='random seed', required=False, type=int)

    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        config = yaml.load(f)
        config = edict(config)

    return args, config


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)

def main():
    # 设置随机数种子
    args, config = parse_args()

    logger, final_output_dir, tb_log_dir = \
        utils.create_logger(config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    args.seed = np.random.randint(0, 10000)
    setup_seed(args.seed)
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    model = get_face_alignment_net(config, stem=args.stem, scbe=False)

    # copy model files
    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.GPUID)
    if torch.cuda.is_available:
        device = torch.device('cuda:{}'.format(0))
    else:
        device = torch.device('cpu:0')

    model = model.to(device)

    # loss
    criterion = torch.nn.MSELoss().to(device)

    optimizer = utils.get_optimizer(config, model)

    best_nme = 100
    last_epoch = config.TRAIN.BEGIN_EPOCH
    if isinstance(config.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch - 1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch - 1
        )

    if args.resume and config.CHECKPOINT_PATH != '':
        checkpoint = torch.load(config.CHECKPOINT_PATH, map_location='cpu')
        last_epoch = checkpoint['epoch']
        best_nme = checkpoint['best_nme']
        current_nme = checkpoint['current_nme']
        best_epoch = checkpoint['best_epoch']
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint (epoch {})"
              .format(checkpoint['epoch']))
        print('best epoch: {}, best nme: {:.4f}, current nme: {:.4f}'.format(best_epoch, best_nme, current_nme))
    else:
        print("=> no checkpoint found")

    dataset_type = get_dataset(config)

    train_loader = DataLoader(
        dataset=dataset_type(config,
                             is_train=True),
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY)

    val_loader = DataLoader(
        dataset=dataset_type(config,
                             is_train=False),
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    best_epoch = 0
    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):

        print('current learning rate: {}'.format(optimizer.param_groups[0]['lr']))
        function.train(config, train_loader, model, criterion,
                       optimizer, epoch, device, writer_dict)

        # evaluate
        nme = function.validate(config, val_loader, model,
                                             criterion, epoch, device, writer_dict)
        lr_scheduler.step()

        is_best = nme < best_nme
        if is_best:
            best_epoch = epoch
        best_nme = min(nme, best_nme)

        print("best:", is_best)
        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        print("best epoch: {}, best_nme: {:.6f}".format(best_epoch, best_nme))
        if is_best:
            torch.save(
                {"state_dict": model.state_dict(),
                 "epoch": epoch + 1,
                 "current_nme": nme,
                 "best_nme": best_nme,
                 "best_epoch": best_epoch,
                 "optimizer": optimizer.state_dict(),
                 "lr_scheduler": lr_scheduler.state_dict(),
                 }, final_output_dir + '/best_checkpoint_{}_{}_{}.pth'.format(args.stem, args.fs_type, args.fusion))
        else:
            torch.save(
                {"state_dict": model.state_dict(),
                 "epoch": epoch + 1,
                 "current_nme": nme,
                 "best_nme": best_nme,
                 "best_epoch": best_epoch,
                 "optimizer": optimizer.state_dict(),
                 "lr_scheduler": lr_scheduler.state_dict(),
                 }, final_output_dir + '/current_checkpoint_{}_{}_{}.pth'.format(args.stem, args.fs_type, args.fusion))

    copyfile(final_output_dir + '/best_checkpoint_{}_{}_{}.pth'.format(args.stem, args.fs_type, args.fusion), final_output_dir + '/nme_{:.4f}_best_checkpoint_{}_{}_{}.pth'.format(best_nme, args.stem, args.fs_type, args.fusion))
    writer_dict['writer'].close()

if __name__ == '__main__':
    main()










