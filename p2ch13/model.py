import math
import random
from collections import namedtuple

import torch
from torch import nn as nn
import torch.nn.functional as F

from util.logconf import logging
from util.unet import UNet

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

class UNetWrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.input_batchnorm = nn.BatchNorm2d(kwargs['in_channels'])
        self.unet = UNet(**kwargs)
        self.final = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        init_set = {
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
            nn.Linear,
        }
        for m in self.modules():
            if type(m) in init_set:
                nn.init.kaiming_normal_(
                    m.weight.data, mode='fan_out', nonlinearity='relu', a=0
                )
                if m.bias is not None:
                    fan_in, fan_out = \
                        nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

        # nn.init.constant_(self.unet.last.bias, -4)
        # nn.init.constant_(self.unet.last.bias, 4)


    def forward(self, input_batch):
        bn_output = self.input_batchnorm(input_batch)
        un_output = self.unet(bn_output)
        fn_output = self.final(un_output)
        return fn_output

class SegmentationAugmentation(nn.Module):
    def __init__(
            self, flip=None, offset=None, scale=None, rotate=None, noise=None
    ):
        super().__init__()

        self.flip = flip
        self.offset = offset
        self.scale = scale
        self.rotate = rotate
        self.noise = noise

    def forward(self, input_g, label_g):
        transform_t = self._build2dTransformMatrix()
        transform_t = transform_t.expand(input_g.shape[0], -1, -1)
        transform_t = transform_t.to(input_g.device, torch.float32)
        affine_t = F.affine_grid(transform_t[:,:2],
                input_g.size(), align_corners=False)

        augmented_input_g = F.grid_sample(input_g,
                affine_t, padding_mode='border',
                align_corners=False)
        augmented_label_g = F.grid_sample(label_g.to(torch.float32),
                affine_t, padding_mode='border',
                align_corners=False)

        if self.noise:
            noise_t = torch.randn_like(augmented_input_g)
            noise_t *= self.noise

            augmented_input_g += noise_t

        return augmented_input_g, augmented_label_g > 0.5

    def _build2dTransformMatrix(self):
        transform_t = torch.eye(3)

        for i in range(2):
            if self.flip:
                if random.random() > 0.5:
                    transform_t[i,i] *= -1

            if self.offset:
                offset_float = self.offset
                random_float = (random.random() * 2 - 1)
                transform_t[2,i] = offset_float * random_float

            if self.scale:
                scale_float = self.scale
                random_float = (random.random() * 2 - 1)
                transform_t[i,i] *= 1.0 + scale_float * random_float

        if self.rotate:
            angle_rad = random.random() * math.pi * 2
            s = math.sin(angle_rad)
            c = math.cos(angle_rad)

            rotation_t = torch.tensor([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]])

            transform_t @= rotation_t

        return transform_t


# MaskTuple = namedtuple('MaskTuple', 'raw_dense_mask, dense_mask, body_mask, air_mask, raw_candidate_mask, candidate_mask, lung_mask, neg_mask, pos_mask')
#
# class SegmentationMask(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         self.conv_list = nn.ModuleList([
#             self._make_circle_conv(radius) for radius in range(1, 8)
#         ])
#
#     def _make_circle_conv(self, radius):
#         diameter = 1 + radius * 2
#
#         a = torch.linspace(-1, 1, steps=diameter)**2
#         b = (a[None] + a[:, None])**0.5
#
#         circle_weights = (b <= 1.0).to(torch.float32)
#
#         conv = nn.Conv2d(1, 1, kernel_size=diameter, padding=radius, bias=False)
#         conv.weight.data.fill_(1)
#         conv.weight.data *= circle_weights / circle_weights.sum()
#
#         return conv
#
#
#     def erode(self, input_mask, radius, threshold=1):
#         conv = self.conv_list[radius - 1]
#         input_float = input_mask.to(torch.float32)
#         result = conv(input_float)
#
#         # log.debug(['erode in ', radius, threshold, input_float.min().item(), input_float.mean().item(), input_float.max().item()])
#         # log.debug(['erode out', radius, threshold, result.min().item(), result.mean().item(), result.max().item()])
#
#         return result >= threshold
#
#     def deposit(self, input_mask, radius, threshold=0):
#         conv = self.conv_list[radius - 1]
#         input_float = input_mask.to(torch.float32)
#         result = conv(input_float)
#
#         # log.debug(['deposit in ', radius, threshold, input_float.min().item(), input_float.mean().item(), input_float.max().item()])
#         # log.debug(['deposit out', radius, threshold, result.min().item(), result.mean().item(), result.max().item()])
#
#         return result > threshold
#
#     def fill_cavity(self, input_mask):
#         cumsum = input_mask.cumsum(-1)
#         filled_mask = (cumsum > 0)
#         filled_mask &= (cumsum < cumsum[..., -1:])
#         cumsum = input_mask.cumsum(-2)
#         filled_mask &= (cumsum > 0)
#         filled_mask &= (cumsum < cumsum[..., -1:, :])
#
#         return filled_mask
#
#
#     def forward(self, input_g, raw_pos_g):
#         gcc_g = input_g + 1
#
#         with torch.no_grad():
#             # log.info(['gcc_g', gcc_g.min(), gcc_g.mean(), gcc_g.max()])
#
#             raw_dense_mask = gcc_g > 0.7
#             dense_mask = self.deposit(raw_dense_mask, 2)
#             dense_mask = self.erode(dense_mask, 6)
#             dense_mask = self.deposit(dense_mask, 4)
#
#             body_mask = self.fill_cavity(dense_mask)
#             air_mask = self.deposit(body_mask & ~dense_mask, 5)
#             air_mask = self.erode(air_mask, 6)
#
#             lung_mask = self.deposit(air_mask, 5)
#
#             raw_candidate_mask = gcc_g > 0.4
#             raw_candidate_mask &= air_mask
#             candidate_mask = self.erode(raw_candidate_mask, 1)
#             candidate_mask = self.deposit(candidate_mask, 1)
#
#             pos_mask = self.deposit((raw_pos_g > 0.5) & lung_mask, 2)
#
#             neg_mask = self.deposit(candidate_mask, 1)
#             neg_mask &= ~pos_mask
#             neg_mask &= lung_mask
#
#             # label_g = (neg_mask | pos_mask).to(torch.float32)
#             label_g = (pos_mask).to(torch.float32)
#             neg_g = neg_mask.to(torch.float32)
#             pos_g = pos_mask.to(torch.float32)
#
#         mask_dict = {
#             'raw_dense_mask': raw_dense_mask,
#             'dense_mask': dense_mask,
#             'body_mask': body_mask,
#             'air_mask': air_mask,
#             'raw_candidate_mask': raw_candidate_mask,
#             'candidate_mask': candidate_mask,
#             'lung_mask': lung_mask,
#             'neg_mask': neg_mask,
#             'pos_mask': pos_mask,
#         }
#
#         return label_g, neg_g, pos_g, lung_mask, mask_dict
