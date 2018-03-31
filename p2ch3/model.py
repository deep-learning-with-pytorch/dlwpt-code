from torch import nn as nn

from util.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

class LunaModel(nn.Module):
    def __init__(self, layer_count, in_channels, conv_channels):
        super().__init__()

        layer_list = []
        for layer_ndx in range(layer_count):
            layer_list += [
                nn.Conv3d(in_channels, conv_channels, kernel_size=3, padding=1, bias=True),
                # nn.BatchNorm3d(conv_channels),
                nn.ReLU(inplace=True),

                nn.Conv3d(conv_channels, conv_channels, kernel_size=3, padding=1, bias=True),
                # nn.BatchNorm3d(conv_channels),
                nn.ReLU(inplace=True),

                nn.MaxPool3d(2, 2),
            ]

            in_channels = conv_channels
            conv_channels *= 2

        self.convAndPool_seq = nn.Sequential(*layer_list)
        self.fullyConnected_layer = nn.Linear(256, 1)


    def forward(self, x):
        conv_out = self.convAndPool_seq(x)
        flattened_out = conv_out.view(conv_out.size(0), -1)

        try:
            classification_out = self.fullyConnected_layer(flattened_out)
        except:
            log.debug(flattened_out.size())
            raise

        return classification_out
