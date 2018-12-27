import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from torchvision import models


from functools import partial

relu = partial(F.relu, inplace=True)

class _ConvBatchNormReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        relu=True,
    ):
        super(_ConvBatchNormReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
        )
        self.add_module(
            "bn",
            nn.BatchNorm2d(
                num_features=out_channels, eps=1e-5, momentum=0.999, affine=True
            ),
        )

        if relu:
            self.add_module("relu", nn.ReLU(True))

    def forward(self, x):
        return super(_ConvBatchNormReLU, self).forward(x)

class ASPP(nn.Module):
    def __init__(self, in_channels):
        super(ASPP, self).__init__()
        pyramids = [6, 12, 18]
        self.stages = nn.Module()
        self.stages.add_module(
            "c0", _ConvBatchNormReLU(in_channels, 256, 1, 1, 0, 1)
        )
        for i, (dilation, padding) in enumerate(zip(pyramids, pyramids)):
            self.stages.add_module(
                "c{}".format(i + 1),
                _ConvBatchNormReLU(in_channels, 256, 3, 1, padding, dilation),
            )
        self.imagepool = nn.Sequential(
            OrderedDict(
                [
                    ("pool", nn.AdaptiveAvgPool2d(1)),
                    ("conv", nn.Conv2d(in_channels, 256, 1, 1, 0, 1)),
                    ("relu", nn.ReLU(True))
                ]
            )
        )
        self.conv1 = _ConvBatchNormReLU(256 * (len(pyramids) + 2), 512, 1, 1, 0, 1)


    def forward(self, x):
        h = self.imagepool(x)
        h = [F.interpolate(h, size=x.shape[2:], mode="bilinear", align_corners=False)]
        for stage in self.stages.children():
            h += [stage(x)]
        h = torch.cat(h, dim=1)
        h = self.conv1(h)
        return h


class Center(nn.Module):
    def __init__(self, channel, type='basic'):
        super(Center, self).__init__()
        if type == 'basic':
            out_channels = [1, 1, 1, 1, 1]
        else:
            out_channels = [2, 4, 8, 16, 1]

        self.dilate_1 = nn.Conv2d(in_channels=out_channels[0] * channel, out_channels=channel, kernel_size=3, stride=1,
                                  dilation=1, padding=1)
        self.relu_1 = relu
        self.dilate_2 = nn.Conv2d(in_channels=out_channels[1] * channel, out_channels=channel, kernel_size=3, stride=1,
                                  dilation=2, padding=2)
        self.relu_2 = relu
        self.dilate_3 = nn.Conv2d(in_channels=out_channels[2] * channel, out_channels=channel, kernel_size=3, stride=1,
                                  dilation=4, padding=4)
        self.relu_3 = relu
        self.dilate_4 = nn.Conv2d(in_channels=out_channels[3] * channel, out_channels=channel, kernel_size=3, stride=1,
                                  dilation=8, padding=8)
        self.relu_4 = relu
        self.dilate_5 = nn.Conv2d(in_channels=out_channels[4] * channel, out_channels=channel, kernel_size=1, stride=1,
                                  dilation=1)

        self.aspp = ASPP(channel)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, i):
        x = self.dilate_1(i)
        x = self.relu_1(x)
        x = self.dilate_2(x)
        x = self.relu_2(x)
        x = self.dilate_3(x)
        x = self.relu_3(x)
        x = self.dilate_4(x)
        x = self.relu_4(x)
        x = self.dilate_5(x)
        aspp = self.aspp(i)
        x += aspp

        return x
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()

        bottleneck_channels = in_channels // 4

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=bottleneck_channels,
                               kernel_size=1, stride=1)
        self.norm1 = nn.BatchNorm2d(bottleneck_channels)
        self.relu1 = relu

        self.deconv2 = nn.ConvTranspose2d(in_channels=bottleneck_channels, out_channels=bottleneck_channels,
                                          kernel_size=3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(bottleneck_channels)
        self.relu2 = relu

        self.conv3 = nn.Conv2d(in_channels=bottleneck_channels, out_channels=out_channels,
                               kernel_size=1, stride=1)
        self.norm3 = nn.BatchNorm2d(out_channels)
        self.relu3 = relu

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)

        return x


class MyNet3(nn.Module):
    def __init__(self, num_classes=1, **kwargs):
        super(MyNet3, self).__init__()
        filters = [64, 128, 256, 512]
        resnet = models.resnet34()

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.center = Center(512)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = relu
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = relu
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.center(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return torch.sigmoid(out)

    def load_pretrained_model(self, path):
        self.load_state_dict(torch.load(path), strict=False)