"""
The BALT module
Created by Sierkinhane(sierkinahne@163.com)
"""
import torch.nn as nn
from lib.models.unet import UNetConv, Up
from torch.nn import Sequential as S

class BALT_MS_2X(nn.Module):
    def __init__(self, num_boundaries=16, num_joints=68, W=48, bilinear=True):
        super(BALT_MS_2X, self).__init__()

        # transition layers
        # _num_tran_channels_to_channels_from_resolution_64x64
        self._1_tran_256_64_64x64 = self._make_transition_layer(256, W, down=False)

        self._2_tran_256_128_64x64 = self._make_transition_layer(256, W*2, down=True)
        self._2_tran_256_128_32x32 = self._make_transition_layer(256, W*2, down=False)

        self._3_tran_256_256_32x32 = self._make_transition_layer(256, W*4, down=True)
        self._3_tran_256_256_16x16 = self._make_transition_layer(256, W*4, down=False)

        self._4_tran_256_512_16x16 = self._make_transition_layer(256, W*8, down=True)
        self._4_tran_256_512_8x8 = self._make_transition_layer(256, W*8, down=False)

        # Boundary to Heatmap part
        self.inconv = UNetConv(num_boundaries, W)
        self.down_1 = S(nn.MaxPool2d(2), UNetConv(W, W * 2), )
        self.down_2 = S(nn.MaxPool2d(2), UNetConv(W * 2, W * 4), )
        self.down_3 = S(nn.MaxPool2d(2), UNetConv(W * 4, W * 8), )
        self.down_4 = S(nn.MaxPool2d(2), UNetConv(W * 8, W * 8), )
        self.up_1 = Up(W * 16, W * 4, bilinear)  # 1024 -->  256
        self.up_2 = Up(W * 8, W * 2, bilinear)   # 512  -->  128
        self.up_3 = Up(W * 4, W * 1, bilinear)   # 256  -->  64
        self.up_4 = Up(W * 2, W * 2, bilinear)   # 128  -->  128
        self.outconv = nn.Conv2d(W * 2, num_joints, kernel_size=1)

        self.init_weights()

    def _make_transition_layer(self, c_in, c_out, down=False):

        if down:
            tran = S(
                nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, bias=False, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            )
        else:
            tran = S(
                nn.Conv2d(c_in, c_out, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            )

        return tran

    def init_weights(self):
        # logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, features):

        # boundary heatmaps to landmark heatmaps part
        x1 = self.inconv(x)
        _1_tran_256_64_64x64 = self._1_tran_256_64_64x64(features[3])
        x1 = x1 + _1_tran_256_64_64x64
        x2 = self.down_1(x1)

        _2_tran_256_128_64x64 = self._2_tran_256_128_64x64(features[3])
        _2_tran_256_128_32x32 = self._2_tran_256_128_32x32(features[2])
        x2 = x2 + _2_tran_256_128_64x64 + _2_tran_256_128_32x32
        x3 = self.down_2(x2)

        _3_tran_256_256_32x32 = self._3_tran_256_256_32x32(features[2])
        _3_tran_256_256_16x16 = self._3_tran_256_256_16x16(features[1])
        x3 = x3 + _3_tran_256_256_32x32 + _3_tran_256_256_16x16
        x4 = self.down_3(x3)

        _4_tran_256_512_16x16 = self._4_tran_256_512_16x16(features[1])
        _4_tran_256_512_8x8 = self._4_tran_256_512_8x8(features[0])
        x4 = x4 + _4_tran_256_512_16x16 + _4_tran_256_512_8x8
        x5 = self.down_4(x4)

        x = self.up_1(x5, x4)
        x = self.up_2(x, x3)
        x = self.up_3(x, x2)
        x = self.up_4(x, x1)
        heatmaps = self.outconv(x)

        return heatmaps