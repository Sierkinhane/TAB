"""
the SCBE module
Created by Sierkinhane(sierkinhane@163.com)
reference: https://github.com/1adrianb/face-alignment.
"""
import torch
import torch.nn as nn
from lib.models.balt import BALT_MS_2X
from lib.models.scbe import FAN

class TAB(nn.Module):
    def __init__(self, stem, num_stacks=2, num_boundaries=16, num_joints=98, W=48):
        super(TAB, self).__init__()

        self.SCBE = FAN(stem, num_stacks=num_stacks, num_boundaries=num_boundaries)
        self.BALT = BALT_MS_2X(num_boundaries=num_boundaries, num_joints=num_joints, W=W, bilinear=True)

    def forward(self, x):

        boundary, features, hourglass_features = self.SCBE(x)
        heatmaps = self.BALT(boundary[-1], hourglass_features[-1])

        return heatmaps


def get_face_alignment_net(cfg, stem='vgg', scbe=True, test=False):

    if scbe:
        model = FAN(stem, num_stacks=cfg.MODEL.NUM_STACKS, num_boundaries=cfg.MODEL.NUM_BOUNDARIES)

    else:
        model = TAB(stem, num_stacks=cfg.MODEL.NUM_STACKS, num_boundaries=cfg.MODEL.NUM_BOUNDARIES, num_joints=cfg.MODEL.NUM_JOINTS, W=cfg.MODEL.WIDTH)

        if not test:
            state_dict = torch.load(cfg.BOUNDARY_DIR + 'best_checkpoint_{}_{}.pth'.format(stem, cfg.DATASET.DATASET), map_location='cpu')
            print('distance:', state_dict['current_dist'])
            model.SCBE.load_state_dict(state_dict['state_dict'])
            for n,p in model.SCBE.named_parameters():
                p.requires_grad = False

    return model

def model_info(model):  # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('\n%5s %50s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%5g %50s %9s %12g %20s %12.3g %12.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))

