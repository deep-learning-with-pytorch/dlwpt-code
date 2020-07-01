import math
import numpy as np

import torch
from torch import nn as nn

from util.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

import random

def augment3d(inp):
    transform_t = torch.eye(4, dtype=torch.float32)
    for i in range(3):
        if True: #'flip' in augmentation_dict:
            if random.random() > 0.5:
                transform_t[i,i] *= -1
        if True: #'offset' in augmentation_dict:
            offset_float = 0.1
            random_float = (random.random() * 2 - 1)
            transform_t[3,i] = offset_float * random_float
    if True:
        angle_rad = random.random() * np.pi * 2
        s = np.sin(angle_rad)
        c = np.cos(angle_rad)

        rotation_t = torch.tensor([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=torch.float32)

        transform_t @= rotation_t
    #print(inp.shape, transform_t[:3].unsqueeze(0).expand(inp.size(0), -1, -1).shape)
    affine_t = torch.nn.functional.affine_grid(
            transform_t[:3].unsqueeze(0).expand(inp.size(0), -1, -1).cuda(),
            inp.shape,
            align_corners=False,
        )

    augmented_chunk = torch.nn.functional.grid_sample(
            inp,
            affine_t,
            padding_mode='border',
            align_corners=False,
        )
    if False: #'noise' in augmentation_dict:
        noise_t = torch.randn_like(augmented_chunk)
        noise_t *= augmentation_dict['noise']

        augmented_chunk += noise_t
    return augmented_chunk


class LunaModel(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8):
        super().__init__()

        self.tail_batchnorm = nn.BatchNorm3d(1)

        self.block1 = LunaBlock(in_channels, conv_channels)
        self.block2 = LunaBlock(conv_channels, conv_channels * 2)
        self.block3 = LunaBlock(conv_channels * 2, conv_channels * 4)
        self.block4 = LunaBlock(conv_channels * 4, conv_channels * 8)

        # self.head_linear = nn.Linear(1152, 2)
        self.head_linear = nn.Linear(1152, 2)
        self.head_activation = nn.Softmax(dim=1)

        self._init_weights()

    # see also https://github.com/pytorch/pytorch/issues/18182
    def _init_weights(self):
        for m in self.modules():
            if type(m) in {
                nn.Linear,
                nn.Conv3d,
                nn.Conv2d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
            }:
                nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    fan_in, fan_out = \
                        nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)


    def forward(self, input_batch):
        bn_output = self.tail_batchnorm(input_batch)

        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        block_out = self.block4(block_out)

        conv_flat = block_out.view(
            block_out.size(0),
            -1,
        )
        linear_output = self.head_linear(conv_flat)

        return linear_output, self.head_activation(linear_output)


class LunaBlock(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels, conv_channels, kernel_size=3, padding=1, bias=True
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            conv_channels, conv_channels, kernel_size=3, padding=1, bias=True
        )
        self.relu2 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool3d(2, 2)


    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        block_out = self.conv2(block_out)
        block_out = self.relu2(block_out)

        return self.maxpool(block_out)
