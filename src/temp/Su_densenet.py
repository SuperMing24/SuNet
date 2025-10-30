import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleneckLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, expansion_ratio):
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


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, compression_factor):
        """
        Args:
            in_channels: inputs relative to transition
            compression_factor: compress channels not only spatial dimensions(height, width) of feature maps in the rate range (0, 1),
                              referred to as DenseNet-C
        """
        super(TransitionLayer, self).__init__()

        # BN -> ReLU -> 1*1 conv -> 2*2 pool
        self.norm = nn.BatchNorm2d(in_channels)
        out_channels = int(in_channels * compression_factor)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)  # 添加 bias=False
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)  # 论文中使用平均池化

    def forward(self, x):
        x = self.conv(F.relu(self.norm(x)))
        out = self.pool(x)
        return out

class DenseNet(nn.Module):
    def __init__(self, denseblock_config, in_channels=3, init_feature_channels=64,
                 growth_rate=32, expansion_ratio=4, compression_factor=0.5, num_classes=1000, **kwargs):
        """
        Args:
            denseblock_config: architecture of densenet denoting the number of layers per denseblock
            in_channels: the number of the initial input channels
            init_feature_channels: the number of the initial output channels
            growth_rate: uniformly set the growth rate of output channels for feature maps in each block throughout the entire DenseNet
            expansion_ratio: uniformly set the output channels scaling in the bottleneck
            compression_factor: the compression ratio uniformly applied to channels by the transition-layer
            num_classes: the number of classifications
            **kwargs: additional flexible parameters
        """
        super(DenseNet, self).__init__()

        # 处理kwargs中的额外参数
        self.dropout_rate = kwargs.get('dropout_rate', 0.0)     # merely give it a try

        # begin with the 7*7 convolutional layer
        self.init_features = nn.Sequential(
            nn.Conv2d(in_channels, init_feature_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(init_feature_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # 创建多个Dense块和转换层
        num_feature_channels = init_feature_channels
        self.dense_blocks = nn.ModuleList()
        self.trans_layers = nn.ModuleList()

        # 根据模型配置数值构建DenseBlock和TransitionLayer
        for i, num_layer in enumerate(denseblock_config):
            current_denseblock = DenseBlock(num_feature_channels, num_layer, growth_rate, expansion_ratio)
            self.dense_blocks.append(current_denseblock)

            # 更新通道数，因需重新计算得出，而非参数传出
            num_feature_channels = num_feature_channels + num_layer * growth_rate

            if i != len(denseblock_config) - 1:
                current_translayer = TransitionLayer(num_feature_channels, compression_factor)
                self.trans_layers.append(current_translayer)
                # 更新通道数：经过TransitionLayer后，通道数被压缩
                num_feature_channels = int(num_feature_channels * compression_factor)

        self.norm_final = nn.BatchNorm2d(num_feature_channels)
        self.classifier = nn.Linear(num_feature_channels, num_classes)

        # 权重初始化
        self.initial_weights()

    def forward(self, x):
        x = self.init_features(x)
        for i in range(len(self.dense_blocks)):
            x = self.dense_blocks[i](x)
            if i < len(self.trans_layers):
                x = self.trans_layers[i](x)

        x = F.relu(self.norm_final(x))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        out = self.classifier(x)
        return out


class SimpleDenseNet(nn.Module):
    """用于快速调试的简化版本"""

    def __init__(self, num_classes=100):
        super().__init__()

        # 极简初始特征提取
        self.init_features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # 减小通道数
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 32->16
        )

        # 单个小型DenseBlock
        self.dense_block = DenseBlock(
            in_channels=16,
            nums_layer=3,  # 只有3层
            growth_rate=8,  # 较小的增长率
            expansion_ratio=2  # 较小的扩展比
        )

        # 最终分类
        self.norm_final = nn.BatchNorm2d(16 + 3 * 8)  # 16 + 24 = 40
        self.classifier = nn.Linear(40, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.init_features(x)
        x = self.dense_block(x)
        x = F.relu(self.norm_final(x))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.classifier(x)

def densenet121(**kwargs):
    return DenseNet(denseblock_config=(6, 12, 24, 16), **kwargs)

