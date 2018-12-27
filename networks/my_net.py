import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
models.resnet34()

from functools import partial

relu = partial(F.relu, inplace=True)

class Center(nn.Module):

    def __init__(self, channel, type='basic'):
        super(Center, self).__init__()
        if type=='basic':
            out_channels = [1, 1, 1, 1, 1]
        else:
            out_channels = [2, 4, 8, 16, 1]
        self.dilate_1 = nn.Conv2d(in_channels=out_channels[0]*channel, out_channels=channel, kernel_size=3, stride=1, dilation=1, padding=1)
        self.relu_1   = relu
        self.dilate_2 = nn.Conv2d(in_channels=out_channels[1]*channel, out_channels=channel, kernel_size=3, stride=1, dilation=2, padding=2)
        self.relu_2   = relu
        self.dilate_3 = nn.Conv2d(in_channels=out_channels[2]*channel, out_channels=channel, kernel_size=3, stride=1, dilation=4, padding=4)
        self.relu_3   = relu
        self.dilate_4 = nn.Conv2d(in_channels=out_channels[3]*channel, out_channels=channel, kernel_size=3, stride=1, dilation=8, padding=8)
        self.relu_4   = relu
        self.dilate_5 = nn.Conv2d(in_channels=out_channels[4]*channel, out_channels=channel, kernel_size=1, stride=1, dilation=1, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.dilate_1(x)
        x = self.relu_1(x)
        x = self.dilate_2(x)
        x = self.relu_2(x)
        x = self.ditate_3(x)
        x = self.relu_3(x)
        x = self.dilate_4(x)
        x = self.relu_4(x)
        x = self.dilate_5(x)

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

        x = self.convtranspose_2(x)
        x = self.norm2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)

        return x

class MyNet(nn.Module):
    def __init__(self, num_classes=1):
        super(MyNet, self).__init__()
        filters= [64, 128, 256, 512]
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





