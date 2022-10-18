import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as S
import torch

"""
clear UNet implementation.
"""

BN_MOMEMTUM = 0.01
class UNetConv(nn.Module):
    """
    conv-bn-relu-conv-bn-relu
    """

    def __init__(self, c_in, c_out):
        super(UNetConv, self).__init__()
        self.UConv = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out, momentum=BN_MOMEMTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out, momentum=BN_MOMEMTUM),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.UConv(x)

class Up(nn.Module):
    """
    Upscaling then double conv(implemented by https://github.com/milesial/Pytorch-UNet)
    """

    def __init__(self, c_in, c_out, bilinear=True):
        super(Up, self).__init__()

        self.conv = UNetConv(c_in, c_out)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, c_in, n_classes, bilinear=True):
        super(UNet, self).__init__()

        c_base = 16
        self.inconv = UNetConv(c_in, c_base)
        self.down_1 = S(nn.MaxPool2d(2), UNetConv(c_base, c_base * 2), )
        self.down_2 = S(nn.MaxPool2d(2), UNetConv(c_base * 2, c_base * 4), )
        self.down_3 = S(nn.MaxPool2d(2), UNetConv(c_base * 4, c_base * 8), )
        self.down_4 = S(nn.MaxPool2d(2), UNetConv(c_base * 8, c_base * 8), )
        self.up_1 = Up(c_base * 16, c_base * 4, bilinear)
        self.up_2 = Up(c_base * 8, c_base * 2, bilinear)
        self.up_3 = Up(c_base * 4, c_base * 1, bilinear)
        self.up_4 = Up(c_base * 2, c_base * 1, bilinear)
        self.outconv = nn.Conv2d(c_base, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inconv(x)
        x2 = self.down_1(x1)
        x3 = self.down_2(x2)
        x4 = self.down_3(x3)
        x5 = self.down_4(x4)
        x = self.up_1(x5, x4)
        x = self.up_2(x, x3)
        x = self.up_3(x, x2)
        x = self.up_4(x, x1)
        logits = self.outconv(x)
        return logits
