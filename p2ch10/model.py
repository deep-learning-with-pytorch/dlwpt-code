
import torch
from torch import nn as nn

from util.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

class LunaModel(nn.Module):
    def __init__(self, layer_count=4, in_channels=1, conv_channels=8):
        super().__init__()

        layer_list = []
        for layer_ndx in range(layer_count):
            layer_list += [
                nn.Conv3d(in_channels, conv_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(conv_channels), # eli: will assume that p1ch6 doesn't use this
                nn.LeakyReLU(inplace=True), # eli: will assume plan ReLU
                nn.Dropout3d(p=0.2),  # eli: will assume that p1ch6 doesn't use this

                nn.Conv3d(conv_channels, conv_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(conv_channels),
                nn.LeakyReLU(inplace=True),
                nn.Dropout3d(p=0.2),

                nn.MaxPool3d(2, 2),
 # tag::model_init[]
           ]

            in_channels = conv_channels
            conv_channels *= 2

        self.convAndPool_seq = nn.Sequential(*layer_list)
        self.fullyConnected_layer = nn.Linear(512, 1)
        self.final = nn.Hardtanh(min_val=0.0, max_val=1.0)


    def forward(self, input_batch):
        conv_output = self.convAndPool_seq(input_batch)
        conv_flat = conv_output.view(conv_output.size(0), -1)

        try:
            classifier_output = self.fullyConnected_layer(conv_flat)
        except:
            log.debug(conv_flat.size())
            raise

        classifier_output = self.final(classifier_output)
        return classifier_output
