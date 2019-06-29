import torch
from torch import nn as nn

from util.logconf import logging
from util.unet import UNet

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

# torch.backends.cudnn.enabled = False

class UNetWrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.batchnorm = nn.BatchNorm2d(kwargs['in_channels'])
        self.unet = UNet(**kwargs)
        self.hardtanh = nn.Hardtanh(min_val=0, max_val=1)

    def forward(self, input):
        bn_output = self.batchnorm(input)
        un_output = self.unet(bn_output)
        ht_output = self.hardtanh(un_output)

        return ht_output



class Simple2dSegmentationModel(nn.Module):
    def __init__(self, layers, in_channels, conv_channels, final_channels):
        super().__init__()
        self.layers = layers

        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.final_channels = final_channels

        layer_list = [
            nn.Conv2d(self.in_channels, self.conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.conv_channels),
            # nn.GroupNorm(1, self.conv_channels),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(inplace=True),
        ]

        for i in range(self.layers):
            layer_list.extend([
                nn.Conv2d(self.conv_channels, self.conv_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.conv_channels),
                # nn.GroupNorm(1, self.conv_channels),
                # nn.ReLU(inplace=True),
                nn.LeakyReLU(inplace=True),
            ])

        layer_list.extend([
            nn.Conv2d(self.conv_channels, self.final_channels, kernel_size=1, bias=True),
            nn.Hardtanh(min_val=0, max_val=1),
        ])

        self.layer_seq = nn.Sequential(*layer_list)


    def forward(self, in_data):
        return self.layer_seq(in_data)


class Dense2dSegmentationModel(nn.Module):
    def __init__(self, layers, input_channels, conv_channels, bottleneck_channels, final_channels):
        super().__init__()
        self.layers = layers

        self.input_channels = input_channels
        self.conv_channels = conv_channels
        self.bottleneck_channels = bottleneck_channels
        self.final_channels = final_channels

        self.layer_list = nn.ModuleList()

        for i in range(layers):
            self.layer_list.append(
                Dense2dSegmentationBlock(
                    input_channels + bottleneck_channels * i,
                    conv_channels,
                    bottleneck_channels,
                )
            )

        self.layer_list.append(
            Dense2dSegmentationBlock(
                input_channels + bottleneck_channels * layers,
                conv_channels,
                bottleneck_channels,
                final_channels,
            )
        )

        self.htanh_layer = nn.Hardtanh(min_val=0, max_val=1)

    def forward(self, input_tensor):
        concat_list = [input_tensor]
        for layer_block in self.layer_list:
            layer_output = layer_block(torch.cat(concat_list, dim=1))
            concat_list.append(layer_output)

        return self.htanh_layer(concat_list[-1])


class Dense2dSegmentationBlock(nn.Module):
    def __init__(self, input_channels, conv_channels, bottleneck_channels, final_channels=None):
        super().__init__()

        self.input_channels = input_channels
        self.conv_channels = conv_channels
        self.bottleneck_channels = bottleneck_channels
        self.final_channels = final_channels or bottleneck_channels

        self.conv1_seq = nn.Sequential(
            nn.Conv2d(self.input_channels, self.bottleneck_channels, kernel_size=1),
            nn.Conv2d(self.bottleneck_channels, self.conv_channels, kernel_size=3, padding=1),
            nn.Conv2d(self.conv_channels, self.bottleneck_channels, kernel_size=1),
            # nn.BatchNorm2d(self.conv_channels),
            nn.GroupNorm(1, self.bottleneck_channels),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(inplace=True),
        )

        self.conv2_seq = nn.Sequential(
            nn.Conv2d(self.input_channels + self.bottleneck_channels, self.bottleneck_channels, kernel_size=1),
            nn.Conv2d(self.bottleneck_channels, self.conv_channels, kernel_size=3, padding=1),
            nn.Conv2d(self.conv_channels, self.final_channels, kernel_size=1),
            # nn.BatchNorm2d(self.conv_channels),
            nn.GroupNorm(1, self.final_channels),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, input_tensor):
        conv1_tensor = self.conv1_seq(input_tensor)
        conv2_tensor = self.conv2_seq(torch.cat([input_tensor, conv1_tensor], dim=1))

        return conv2_tensor


class SegmentationModel(nn.Module):
    def __init__(self, depth, in_channels, tail_channels=None, out_channels=None, final_channels=None):
        super().__init__()
        self.depth = depth

        # self.in_size = in_size
        # self.tailOut_size = in_size #self.in_size - 4
        # self.headIn_size = in_size #None
        # self.out_size = in_size #None

        self.in_channels = in_channels
        self.tailOut_channels = tail_channels or in_channels * 2
        self.headIn_channels = None
        self.out_channels = out_channels or self.tailOut_channels
        self.final_channels = final_channels

        # assert in_size % 2 == 0, repr([in_size, depth])

        self.tail_seq = nn.Sequential(
            nn.ReplicationPad3d(2),
            nn.Conv3d(self.in_channels, self.tailOut_channels, 3),
            nn.GroupNorm(1, self.tailOut_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.tailOut_channels, self.tailOut_channels, 3),
            nn.GroupNorm(1, self.tailOut_channels),
            nn.ReLU(inplace=True),
        )

        if depth:
            self.downsample_layer = nn.MaxPool3d(kernel_size=2, stride=2)
            self.child_layer = SegmentationModel(depth - 1, self.tailOut_channels)

            self.headIn_channels = self.in_channels + self.tailOut_channels + self.child_layer.out_channels
            # self.headIn_size = self.child_layer.out_size * 2
            # self.out_size = self.headIn_size #- 4

            # self.upsample_layer = nn.Upsample(scale_factor=2, mode='trilinear')
        else:
            self.downsample_layer = None
            self.child_layer = None
            # self.upsample_layer = None

            self.headIn_channels = self.in_channels + self.tailOut_channels
            # self.headIn_size = self.tailOut_size
            # self.out_size = self.headIn_size #- 4

        self.head_seq = nn.Sequential(
            nn.ReplicationPad3d(2),
            nn.Conv3d(self.headIn_channels, self.out_channels, 3),
            nn.GroupNorm(1, self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.out_channels, self.out_channels, 3),
            nn.GroupNorm(1, self.out_channels),
            nn.ReLU(inplace=True),
        )

        if self.final_channels:
            self.final_seq = nn.Sequential(
                nn.ReplicationPad3d(1),
                nn.Conv3d(self.out_channels, self.final_channels, 1),
            )
        else:
            self.final_seq = None

    def forward(self, in_data):

        assert in_data.is_contiguous()

        try:
            tail_out = self.tail_seq(in_data)
        except:
            log.debug([in_data.size()])
            raise

        if self.downsample_layer:
            down_out = self.downsample_layer(tail_out)
            child_out = self.child_layer(down_out)
            # up_out = self.upsample_layer(child_out)

            up_out = nn.functional.interpolate(child_out, scale_factor=2, mode='trilinear')

            # crop_int = (tail_out.size(-1) - up_out.size(-1)) // 2
            # crop_out = tail_out[:, :, crop_int:-crop_int, crop_int:-crop_int, crop_int:-crop_int]
            # combined_out = torch.cat([crop_out, up_out], 1)

            combined_out = torch.cat([in_data, tail_out, up_out], 1)
        else:
            combined_out = torch.cat([in_data, tail_out], 1)

        head_out = self.head_seq(combined_out)

        if self.final_seq:
            final_out = self.final_seq(head_out)
            return final_out
        else:
            return head_out


class DenseSegmentationModel(nn.Module):
    def __init__(self, depth, in_channels, conv_channels, final_channels=None):
        super().__init__()
        self.depth = depth

        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.final_channels = final_channels

        self.convA_seq = nn.Sequential(
            nn.Conv3d(self.in_channels, self.conv_channels // 4, 1),
            nn.ReplicationPad3d(1),
            nn.Conv3d(self.conv_channels // 4, self.conv_channels, 3),
            nn.BatchNorm3d(self.conv_channels),
            nn.ReLU(inplace=True),
        )

        self.convB_seq = nn.Sequential(
            nn.Conv3d(self.in_channels + self.conv_channels, self.conv_channels // 4, 1),
            nn.ReplicationPad3d(1),
            nn.Conv3d(self.conv_channels // 4, self.conv_channels, 3),
            nn.BatchNorm3d(self.conv_channels),
            nn.ReLU(inplace=True),
        )

        if self.depth:
            self.downsample_layer = nn.MaxPool3d(kernel_size=2, stride=2)
            self.child_layer = SegmentationModel(depth - 1, self.conv_channels, self.conv_channels * 2)
            self.upsample_layer = nn.Upsample(scale_factor=2, mode='trilinear')

            self.convC_seq = nn.Sequential(
                nn.Conv3d(self.in_channels + self.conv_channels * 3, self.conv_channels // 4, 1),
                nn.ReplicationPad3d(1),
                nn.Conv3d(self.conv_channels // 4, self.conv_channels, 3),
                nn.BatchNorm3d(self.conv_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.downsample_layer = None
            self.child_layer = None
            self.upsample_layer = None

            self.convC_seq = nn.Sequential(
                nn.Conv3d(self.in_channels + self.conv_channels, self.conv_channels // 4, 1),
                nn.ReplicationPad3d(1),
                nn.Conv3d(self.conv_channels // 4, self.conv_channels, 3),
                nn.BatchNorm3d(self.conv_channels),
                nn.ReLU(inplace=True),
            )

        self.convD_seq = nn.Sequential(
            nn.Conv3d(self.in_channels + self.conv_channels, self.conv_channels // 4, 1),
            nn.ReplicationPad3d(1),
            nn.Conv3d(self.conv_channels // 4, self.conv_channels, 3),
            nn.BatchNorm3d(self.conv_channels),
            nn.ReLU(inplace=True),
        )

        if self.final_channels:
            self.final_seq = nn.Sequential(
                # nn.ReplicationPad3d(1),
                nn.Conv3d(self.conv_channels, self.final_channels, 1),
            )
        else:
            self.final_seq = None

    def forward(self, data_in):
        a_out = self.convA_seq(data_in)
        b_out = self.convB_seq(torch.cat([data_in, a_out], 1))

        if self.downsample_layer:
            down_out = self.downsample_layer(b_out)
            child_out = self.child_layer(down_out)
            up_out = self.upsample_layer(child_out)

            c_out = self.convC_seq(torch.cat([data_in, b_out, up_out], 1))
        else:
            c_out = self.convC_seq(torch.cat([data_in, b_out], 1))

        d_out = self.convD_seq(torch.cat([data_in, c_out], 1))

        if self.final_seq:
            return self.final_seq(d_out)
        else:
            return d_out
