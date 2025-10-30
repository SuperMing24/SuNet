import torch
import torch.nn as nn
import torch.nn.functional as F


class FirstConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FirstConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout2d(0.2)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.dropout(self.conv(F.relu(self.bn(x))))
        return x


"""------------------------------------------------------------------------------------------------------------"""
class BottleneckLayer(nn.Module):
    def __init__(self, in_channels, growth_rate=18, expansion_ratio=4):
        """
        Args:
            in_channels: channel size of input features concatenated by all previous layers
            growth_rate: output feature map channels size
            expansion_ratio: output channels scaling via 1*1 convolution layer
        """
        super(BottleneckLayer, self).__init__()
        # 瓶颈结构：BN -> ReLU -> 1x1 Conv -> BN -> ReLU -> 3x3 Conv
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=growth_rate * expansion_ratio,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.norm2 = nn.BatchNorm2d(growth_rate * expansion_ratio)
        self.conv2 = nn.Conv2d(in_channels=growth_rate * expansion_ratio,
                               out_channels=growth_rate,
                               kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        """
        method: concatenate the input and output of feature maps in channel dimension
        """
        out = self.conv1(F.relu(self.norm1(x)))
        out = self.conv2(F.relu(self.norm2(out)))
        out = torch.cat([x, out], 1)
        return out


class DenseBlock(nn.Module):
    def __init__(self, in_channels, nums_layer, growth_rate, expansion_ratio):
        """
        Args:
            in_channels: channels of feature maps in DenseBlock
            nums_layer: number of layers in DenseBlock
            growth_rate: same as BottleneckLayer
            expansion_ratio: same as BottleneckLayer
        """
        super(DenseBlock, self).__init__()

        # BottleneckLayer已经实现每次输出的特征图（在通道维度上）叠加输入的数量，现在控制输入尺寸即可
        self.layers = nn.ModuleList()
        for i in range(nums_layer):
            current_layer = BottleneckLayer(in_channels + growth_rate * i, growth_rate, expansion_ratio)
            self.layers.append(current_layer)

    def forward(self, x):
        """
        method: perform forward pass by applying all layers in the module sequentially
        """
        for layer in self.layers:
            x = layer(x)
        return x


# class TransitionLayer(nn.Module):
#     def __init__(self, in_channels, compression_factor):
#         """
#         Args:
#             in_channels: inputs relative to transition
#             compression_factor: compress channels not only spatial dimensions(height, width) of feature maps in the rate range (0, 1),
#                               referred to as DenseNet-C
#         """
#         super(TransitionLayer, self).__init__()
#
#         # BN -> ReLU -> 1*1 conv -> 2*2 pool
#         self.norm = nn.BatchNorm2d(in_channels)
#         out_channels = int(in_channels * compression_factor)
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)  # 添加 bias=False
#         self.pool = nn.AvgPool2d(kernel_size=2, stride=2)  # 论文中使用平均池化
#
#     def forward(self, x):
#         x = self.conv(F.relu(self.norm(x)))
#         out = self.pool(x)
#         return out
"""------------------------------------------------------------------------------------------------------------"""
# class DoubleConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(DoubleConv, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#
#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         return x



class DownConv(nn.Module):
    def __init__(self, in_channels, nums_layer, growth_rate, expansion_ratio):
        super().__init__()
        self.double_conv = DenseBlock(in_channels=in_channels, nums_layer=nums_layer, growth_rate=growth_rate, expansion_ratio=expansion_ratio)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.double_conv(x)
        return x


def db_out_channel(in_channels, nums_layer, growth_rate=18):
    return in_channels + growth_rate * nums_layer

class UpConv(nn.Module):
    def __init__(self, in_channels, nums_layer, growth_rate, expansion_ratio):
        super(UpConv, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=self.db_out_channel(in_channels,
                                                                                               nums_layer, growth_rate), kernel_size=2, stride=2, padding=0)
        self.double_conv = DenseBlock(in_channels=in_channels, nums_layer=nums_layer, growth_rate=growth_rate, expansion_ratio=expansion_ratio)


    def forward(self, x_front, x_rear):
        x_rear = self.up(x_rear)
        x = torch.cat((x_front, x_rear), dim=1)
        x = self.double_conv(x)
        return x


# denseblock_config=(6, 12, 24, 16)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        self.input_conv = FirstConv(in_channels=in_channels, out_channels=48)
        out_channels = 48
        self.down_convs={}
        for nums_layer in [4,5,7,9,11]:
            DownConv(in_channels=out_channels, nums_layer=nums_layer)
            out_channels = db_out_channel(out_channels, nums_layer)


        #
        # self.down_conv1 = DownConv(in_channels=out_channels, nums_layer=4)
        #
        # out_channels = db_out_channel(out_channels, 4)
        # self.down_conv2 = DownConv(in_channels=out_channels, nums_layer=5)
        # out_channels = db_out_channel(out_channels, 4)
        # self.down_conv2 = DownConv(in_channels=out_channels, nums_layer=5)
        # out_channels = db_out_channel(out_channels, 4)
        # self.down_conv2 = DownConv(in_channels=out_channels, nums_layer=5)



        # self.down_conv3 = DownConv(in_channels=256, out_channels=512)
        # self.down_conv4 = DownConv(in_channels=512, out_channels=1024)

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

        # y5 = F.adaptive_avg_pool2d(y5, (1, 1))
        # y5 = y5.view(y5.size(0), -1)

        return y5
