# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# Modified by Sierkinhane(sierkinhane@163.com)
# ------------------------------------------------------------------------------

import os
import pprint
import argparse

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lib.models.tab import get_face_alignment_net
from lib.utils import utils
from lib.datasets import get_dataset
from lib.core import function
from easydict import EasyDict as edict
import yaml

def parse_args():

    parser = argparse.ArgumentParser(description='Test TAB')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        required=True, type=str)
    parser.add_argument('--stem', default='vgg', help='experiments', type=str)
    parser.add_argument('--best_model', default='', help='the best checkpoint for evaluation', required=True, type=str)
    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        config = yaml.load(f)
        config = edict(config)

    return args, config


def main():

    args, config = parse_args()

    logger, final_output_dir, tb_log_dir = \
        utils.create_logger(config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    model = get_face_alignment_net(config, stem=args.stem, scbe=False, test=True)

    state_dict = torch.load(args.best_model, map_location='cpu')
    if isinstance(state_dict, dict):
        print("nme", state_dict['best_nme'])
        model.load_state_dict(state_dict['state_dict'])
    else:
        model.load_state_dict(state_dict['state_dict'])

    if torch.cuda.is_available:
        device = torch.device('cuda:{}'.format(config.GPUID))
    else:
        device = torch.device('cpu:0')

    model = model.to(device)

    dataset_type = get_dataset(config)

    test_loader = DataLoader(
        dataset=dataset_type(config,
                             is_train=False),
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    if not os.path.exists(config.VIS_VALIDATION_DIR):
        os.mkdir(config.VIS_VALIDATION_DIR)

    flag = config.DATASET.TESTSET[:-4].split('/')[-1].split('_')[-1]
    if not os.path.exists(config.VIS_VALIDATION_DIR + flag):
        os.mkdir(config.VIS_VALIDATION_DIR + flag)

    nme, nme_for_auc = function.inference(config, test_loader, model, device, flag)
    print('{:.4f}'.format(nme))

    from lib.utils.utils import AUCError
    AUCError(nme_for_auc)
    AUCError(nme_for_auc, failureThreshold=0.08)

    ## codes were borrowed from ..
    if config.DATASET.DATASET == '300WLP':
        import numpy as np
        nme = np.array(nme_for_auc, dtype=np.float32)
        yaw_list = np.load(config.DATASET.YAW_LIST)
        yaw_list_abs = np.abs(yaw_list)
        ind_yaw_1 = np.where(yaw_list_abs <= 30)
        ind_yaw_2 = np.where((yaw_list_abs > 30) & (yaw_list_abs <=60))
        ind_yaw_3 = np.where(yaw_list_abs > 60)
        ind_yaw_1 = list(ind_yaw_1[0])
        ind_yaw_2 = list(ind_yaw_2[0])
        ind_yaw_3 = list(ind_yaw_3[0])
        nme_1 = nme[ind_yaw_1]
        nme_2 = nme[ind_yaw_2]
        nme_3 = nme[ind_yaw_3]

        mean_nme_1 = np.mean(nme_1) * 100
        mean_nme_2 = np.mean(nme_2) * 100
        mean_nme_3 = np.mean(nme_3) * 100
        # mean_nme_all = np.mean(nme_list) * 100

        std_nme_1 = np.std(nme_1) * 100
        std_nme_2 = np.std(nme_2) * 100
        std_nme_3 = np.std(nme_3) * 100
        # std_nme_all = np.std(nme_list) * 100

        mean_all = [mean_nme_1, mean_nme_2, mean_nme_3]
        mean = np.mean(mean_all)
        std = np.std(mean_all)

        s1 = '[ 0, 30]\tMean: \x1b[32m{:.3f}\x1b[0m, Std: {:.3f}'.format(mean_nme_1, std_nme_1)
        s2 = '[30, 60]\tMean: \x1b[32m{:.3f}\x1b[0m, Std: {:.3f}'.format(mean_nme_2, std_nme_2)
        s3 = '[60, 90]\tMean: \x1b[32m{:.3f}\x1b[0m, Std: {:.3f}'.format(mean_nme_3, std_nme_3)
        s4 = '[ 0, 90]\tMean: \x1b[31m{:.3f}\x1b[0m, Std: \x1b[31m{:.3f}\x1b[0m'.format(mean, std)
        s = '\n'.join([s1, s2, s3, s4])
        print(s)

if __name__ == '__main__':
    main()

