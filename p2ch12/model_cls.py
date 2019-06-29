import math

import torch.nn as nn

from util.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)


class LunaModel(nn.Module):
    def __init__(self, layer_count=4, in_channels=1, conv_channels=8):
        super().__init__()

        self.input_batchnorm = nn.BatchNorm3d(1)

        layer_list = []
        for layer_ndx in range(layer_count):
            layer_list += [
                nn.Conv3d(in_channels, conv_channels, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(conv_channels, conv_channels, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(2, 2),
            ]

            in_channels = conv_channels
            conv_channels *= 2

        self.convAndPool_seq = nn.Sequential(*layer_list)
        self.fullyConnected_layer = nn.Linear(576, 2)
        self.final = nn.Softmax(dim=1)

        self._init_weights()

    def _init_weights(self):
        # see also https://github.com/pytorch/pytorch/issues/18182
        for m in self.modules():
            if type(m) in {
                nn.Conv2d,
                nn.Conv3d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
                nn.Linear,
            }:
                # log.debug(m)
                # nn.init.kaiming_normal_(m.weight.data, mode='fan_out', a=0)
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.input_batchnorm(input_batch)
        conv_output = self.convAndPool_seq(bn_output)
        conv_flat = conv_output.view(conv_output.size(0), -1)
        classifier_output = self.fullyConnected_layer(conv_flat)

        return classifier_output, self.final(classifier_output)


class AlternateLunaModel(nn.Module):
    def __init__(self, layer_count=4, in_channels=1, conv_channels=64):
        super().__init__()

        layer_list = [nn.BatchNorm3d(1)]
        for layer_ndx in range(layer_count):
            layer_list += [
                nn.Conv3d(in_channels, conv_channels, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(conv_channels, conv_channels // 2, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(2, 2),
            ]

            conv_channels //= 2
            in_channels = conv_channels

        self.convAndPool_seq = nn.Sequential(*layer_list)
        self.fullyConnected_layer = nn.Linear(36, 2)
        self.final = nn.Softmax(dim=1)

        # see also https://github.com/pytorch/pytorch/issues/18182
        for m in self.modules():
            if type(m) in {
                nn.Conv2d,
                nn.Conv3d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
                nn.Linear,
            }:
                # log.debug(m)
                # nn.init.kaiming_normal_(m.weight.data, mode='fan_out', a=0)
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        conv_output = self.convAndPool_seq(input_batch)
        conv_flat = conv_output.view(conv_output.size(0), -1)
        classifier_output = self.fullyConnected_layer(conv_flat)

        return classifier_output, self.final(classifier_output)

