import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from functools import partial

non_linearity = partial(F.relu, inplace=True)


class res_block(nn.Module):
    def __init__(self, in_channels, n_filters, strides):
        super(res_block, self).__init__()

        self.norm_1 = nn.BatchNorm2d(in_channels)
        self.relu_1 = non_linearity
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=n_filters[0],
                                kernel_size=3, stride=strides[0], padding=1)

        self.norm_2 = nn.BatchNorm2d(n_filters[0])
        self.relu_2 = non_linearity
        self.conv_2 = nn.Conv2d(in_channels=n_filters[0], out_channels=n_filters[1],
                                kernel_size=3, stride=strides[1], padding=1)

        self.shortcut = nn.Conv2d(in_channels=in_channels, out_channels=n_filters[1],
                                  kernel_size=1, stride=strides[0])

    def forward(self, input):
        x = self.norm_1(input)
        x = self.relu_1(x)
        x = self.conv_1(x)

        x = self.norm_2(x)
        x = self.relu_2(x)
        x = self.conv_2(x)

        shortcut = self.shortcut(input)
        x = torch.add(x, shortcut)
        return x


class bridge(nn.Module):
    def __init__(self, in_channels, n_filters, strides):
        super(bridge, self).__init__()
        self.norm_1 = nn.BatchNorm2d(in_channels)
        self.relu_1 = non_linearity
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=n_filters[0],
                                kernel_size=3, stride=strides[0], padding=1)

        self.norm_2 = nn.BatchNorm2d(n_filters[0])
        self.relu_2 = non_linearity
        self.conv_2 = nn.Conv2d(in_channels=n_filters[0], out_channels=n_filters[1],
                                kernel_size=3, stride=strides[1], padding=1)

    def forward(self, input):
        x = self.norm_1(input)
        x = self.relu_1(x)
        x = self.conv_1(x)

        x = self.norm_2(x)
        x = self.relu_2(x)
        x = self.conv_2(x)
        return x


class upsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
        super(upsample, self).__init__()
        self.transpose = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                            kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.relu = non_linearity

    def forward(self, x):
        x = self.transpose(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class Residual_Unet(nn.Module):
    def __init__(self, num_classes=1, in_channels=3, **kwargs):
        super(Residual_Unet, self).__init__()

        self.first_conv = nn.Conv2d(in_channels=in_channels, out_channels=8,
                                    kernel_size=3, stride=1, padding=1)
        self.first_norm = nn.BatchNorm2d(num_features=8)
        self.first_relu = non_linearity
        self.second_conv = nn.Conv2d(in_channels=8, out_channels=8,
                                     kernel_size=3, stride=1, padding=1)

        self.first_shortcut = nn.Conv2d(in_channels=in_channels, out_channels=8,
                                        kernel_size=1, stride=1)

        self.encoder_1 = res_block(in_channels=8, n_filters=[16, 16], strides=[2, 1])
        self.encoder_2 = res_block(in_channels=16, n_filters=[32, 32], strides=[2, 1])
        self.encoder_3 = res_block(in_channels=32, n_filters=[64, 64], strides=[2, 1])
        self.encoder_4 = res_block(in_channels=64, n_filters=[128, 128], strides=[2, 1])
        self.encoder_5 = res_block(in_channels=128, n_filters=[256, 256], strides=[2, 1])

        # ******************************************************************************************
        self.bridge = bridge(in_channels=256, n_filters=[512, 512], strides=[2, 1])
        #******************************************************************************************

        self.deconv_6 = upsample(in_channels=512, out_channels=256,
                                 kernel_size=3, stride=2, padding=1, output_padding=1)
        self.res_block_6 = res_block(in_channels=512, n_filters=[256, 256], strides=[1, 1])
        # ******************************************************************************************
        self.deconv_5 = upsample(in_channels=256, out_channels=128,
                                 kernel_size=3, stride=2, padding=1, output_padding=1)
        self.res_block_5 = res_block(in_channels=256, n_filters=[128, 128], strides=[1, 1])
        # ******************************************************************************************
        self.deconv_4 = upsample(in_channels=128, out_channels=64,
                                 kernel_size=3, stride=2, padding=1, output_padding=1)
        self.res_block_4 = res_block(in_channels=128, n_filters=[64, 64], strides=[1, 1])
        # ******************************************************************************************
        self.deconv_3 = upsample(in_channels=64, out_channels=32,
                                 kernel_size=3, stride=2, padding=1, output_padding=1)
        self.res_block_3 = res_block(in_channels=64, n_filters=[32, 32], strides=[1, 1])
        # ******************************************************************************************
        self.deconv_2 = upsample(in_channels=32, out_channels=16,
                                 kernel_size=3, stride=2, padding=1, output_padding=1)
        self.res_block_2 = res_block(in_channels=32, n_filters=[16, 16], strides=[1, 1])
        # ******************************************************************************************
        self.deconv_1 = upsample(in_channels=16, out_channels=8,
                                 kernel_size=3, stride=2, padding=1, output_padding=1)
        self.res_block_1 = res_block(in_channels=16, n_filters=[8, 8], strides=[1, 1])

        self.final_conv = nn.Conv2d(in_channels=8, out_channels=1,
                                    kernel_size=1, stride=1)

    def forward(self, input):
        # encoder
        x = self.first_conv(input)
        x = self.first_norm(x)
        x = self.first_relu(x)
        x = self.second_conv(x)

        shortcut = self.first_shortcut(input)
        e1 = torch.add(x, shortcut)

        e2 = self.encoder_1(e1)
        e3 = self.encoder_2(e2)
        e4 = self.encoder_3(e3)
        e5 = self.encoder_4(e4)
        e6 = self.encoder_5(e5)

        # bridge
        b = self.bridge(e6)

        # decoder

        d6 = self.deconv_6(b)
        d6 = torch.cat((e6, d6), 1)
        d6 = self.res_block_6(d6)

        d5 = self.deconv_5(d6)
        d5 = torch.cat((e5, d5), 1)
        d5 = self.res_block_5(d5)

        d4 = self.deconv_4(d5)
        d4 = torch.cat((e4, d4), 1)
        d4 = self.res_block_4(d4)

        d3 = self.deconv_3(d4)
        d3 = torch.cat((e3, d3), 1)
        d3 = self.res_block_3(d3)

        d2 = self.deconv_2(d3)
        d2 = torch.cat((e2, d2), 1)
        d2 = self.res_block_2(d2)

        d1 = self.deconv_1(d2)
        d1 = torch.cat((e1, d1), 1)
        d1 = self.res_block_1(d1)

        x = self.final_conv(d1)

        return torch.sigmoid(x)


if __name__ == '__main__':
    net = Residual_Unet()
    n = 0
    for param in net.parameters():
        i = 1
        for x in param.size():
            i = i*x
        n = n+ i
    print n