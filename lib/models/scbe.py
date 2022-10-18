"""
the SCBE module
Created by Sierkinhane(sierkinhane@163.com)
reference: https://github.com/1adrianb/face-alignment.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as S
from torchvision import models

def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=strd, padding=padding, bias=bias)

class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3

class HourGlass(nn.Module):
    def __init__(self, num_modules, depth, num_features):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features
        self.multi_scale_features = []
        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(self.features, self.features))

        self.add_module('b2_' + str(level), ConvBlock(self.features, self.features))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(self.features, self.features))

        self.add_module('b3_' + str(level), ConvBlock(self.features, self.features))

    def _forward(self, level, inp):

        # Upper branch
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)

        # Lower branch
        low1 = F.avg_pool2d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)

        up2 = F.interpolate(low3, scale_factor=2, mode='nearest')

        add = up1 + up2
        self.multi_scale_features.append(add)

        return add

    def forward(self, x):
        x = self._forward(self.depth, x)
        msf = self.multi_scale_features
        self.multi_scale_features = []
        return x, msf


class FAN(nn.Module):
    def __init__(self, sl_type, num_stacks=2, num_boundaries=68):
        super(FAN, self).__init__()
        self.num_stacks = num_stacks
        self.sl_type = sl_type
        # Stacking part for boundary estimation
        for hg_module in range(self.num_stacks):
            self.add_module('m' + str(hg_module), HourGlass(1, 4, 256))
            self.add_module('top_m_' + str(hg_module), ConvBlock(256, 256))
            self.add_module('conv_last' + str(hg_module),
                            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
            self.add_module('bn_end' + str(hg_module), nn.BatchNorm2d(256))
            self.add_module('l' + str(hg_module), nn.Conv2d(256,
                                                            num_boundaries, kernel_size=1, stride=1, padding=0))

            if hg_module < self.num_stacks - 1:
                self.add_module(
                    'bl' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                self.add_module('al' + str(hg_module), nn.Conv2d(num_boundaries,
                                                                 256, kernel_size=1, stride=1, padding=0))

        self.init_weights()

        self.features = self._get_stem_layers()


    def _get_stem_layers(self):

        if self.sl_type == 'vgg':
            vgg_pretrained_features = models.vgg16_bn(pretrained=True).features
            features = S()
            for x in range(23):
                features.add_module(str(x), vgg_pretrained_features[x])
        elif self.sl_type == "resnet":
            resnet_pretrained_features = models.resnet152(pretrained=True)
            features = S(
                resnet_pretrained_features.conv1,
                resnet_pretrained_features.bn1,
                resnet_pretrained_features.relu,
                resnet_pretrained_features.maxpool,
                resnet_pretrained_features.layer1,
            )
        elif self.sl_type == "base":
            features = S(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                ConvBlock(64, 128),
                nn.MaxPool2d(2),
                ConvBlock(128, 128),
                ConvBlock(128, 256),
            )

        return features

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.features(x)
        # raw = x
        previous = x

        outputs = []
        features = []
        hourglass_features = []
        features.append(x)
        for i in range(self.num_stacks):
            hg, msf = self._modules['m' + str(i)](previous)
            hourglass_features.append(msf)
            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)

            ll = F.relu(self._modules['bn_end' + str(i)]
                        (self._modules['conv_last' + str(i)](ll)), True)
            # Predict heatmaps
            tmp_out = self._modules['l' + str(i)](ll)
            outputs.append(tmp_out)
            features.append(ll)

            if i < self.num_stacks - 1:
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll + tmp_out_

        return outputs, features, hourglass_features

