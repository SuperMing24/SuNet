import os
# import sys
# sys.path.append('./src_old')

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = DoubleConv(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x):
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.double_conv(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2,
                                     padding=0)
        self.double_conv = DoubleConv(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x_front, x_rear):
        x_rear = self.up(x_rear)
        x = torch.cat((x_front, x_rear), dim=1)
        x = self.double_conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        self.input_conv = DoubleConv(in_channels=in_channels, out_channels=64)
        self.down_conv1 = DownConv(in_channels=64, out_channels=128)
        self.down_conv2 = DownConv(in_channels=128, out_channels=256)
        self.down_conv3 = DownConv(in_channels=256, out_channels=512)
        self.down_conv4 = DownConv(in_channels=512, out_channels=1024)

        self.up_conv1 = UpConv(in_channels=1024, out_channels=512)
        self.up_conv2 = UpConv(in_channels=512, out_channels=256)
        self.up_conv3 = UpConv(in_channels=256, out_channels=128)
        self.up_conv4 = UpConv(in_channels=128, out_channels=64)
        self.output_conv = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.input_conv(x)
        x2 = self.down_conv1(x1)
        x3 = self.down_conv2(x2)
        x4 = self.down_conv3(x3)
        x5 = self.down_conv4(x4)

        y1 = self.up_conv1(x4, x5)
        y2 = self.up_conv2(x3, y1)
        y3 = self.up_conv3(x2, y2)
        y4 = self.up_conv4(x1, y3)
        y5 = self.output_conv(y4)

        return y5

