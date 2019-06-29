import math

from torch import nn as nn

from util.logconf import logging
from util.unet import UNet

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

class UNetWrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.batchnorm = nn.BatchNorm2d(kwargs['in_channels'])
        self.unet = UNet(**kwargs)
        self.final = nn.Sigmoid()

        for m in self.modules():
            if type(m) in {
                nn.Conv2d,
                nn.Conv3d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
                nn.Linear,
            }:
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='leaky_relu', a=0)
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)


    def forward(self, input):
        bn_output = self.batchnorm(input)
        un_output = self.unet(bn_output)
        fn_output = self.final(un_output)

        return fn_output
