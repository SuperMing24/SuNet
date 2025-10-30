import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseLayer(nn.Module):
    """密集层内部结构，控制特征图的增长"""
    def __init__(self, in_channels, growth_rate, dropout_rate=0.2):
        super(DenseLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, growth_rate,
                             kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = self.dropout(out)
        return out

class DenseBlock(nn.Module):
    """密集块"""

    def __init__(self, num_layers, in_channels, growth_rate, dropout_rate=0.2):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            layer = DenseLayer(in_channels + i * growth_rate,
                               growth_rate, dropout_rate)
            self.layers.append(layer)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        return torch.cat(features, 1)


class SerialDenseBlock(nn.Module):
    """串行密集块，用于解码器（输入不与输出拼接）"""

    def __init__(self, num_layers, in_channels, growth_rate, dropout_rate=0.2):
        super(SerialDenseBlock, self).__init__()
        self.layers = nn.ModuleList()

        current_channels = in_channels
        for i in range(num_layers):
            # 每层只接收前一层的输出，不进行密集连接
            layer = DenseLayer(current_channels, growth_rate, dropout_rate)
            self.layers.append(layer)
            current_channels = growth_rate

    def forward(self, x):
        # 串行连接：每层的输出作为下一层的输入
        for layer in self.layers:
            x = layer(x)
        return x

class TransitionDown(nn.Module):
    """下采样过渡层，按照论文Table1实现"""
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super(TransitionDown, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels,
                             kernel_size=1, bias=False)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = self.dropout(out)
        out = self.pool(out)
        return out

class TransitionUp(nn.Module):
    """上采样过渡层，按照论文Table1实现"""
    def __init__(self, in_channels, out_channels):
        super(TransitionUp, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=3,
            stride=2, padding=1, output_padding=1
        )
    def forward(self, x):
        return self.conv_transpose(x)


class FCDenseNet103(nn.Module):
    """FC-DenseNet103"""

    def __init__(self, in_channels=3, num_classes=21, growth_rate=16, dropout_rate=0.2):
        super(FCDenseNet103, self).__init__()

        # 论文里是这样写的
        encoder_layers = [4, 5, 7, 10, 12]
        decoder_layers = [12, 10, 7, 5, 4]

        self.initial_conv = nn.Conv2d(in_channels, 48, kernel_size=3, padding=1, bias=False)
        self.encoder_blocks = nn.ModuleList()
        self.transition_downs = nn.ModuleList()

        current_channels = 48
        skip_connections_channels = []

        # 构建编码器路径
        for i, num_layers in enumerate(encoder_layers):
            # 密集块
            dense_block = DenseBlock(num_layers, current_channels, growth_rate, dropout_rate)
            self.encoder_blocks.append(dense_block)

            # 更新通道数
            current_channels += num_layers * growth_rate
            skip_connections_channels.append(current_channels)

            # 过渡下采样（最后一个块后不下采样）
            if i < len(encoder_layers) - 1:
                # 论文中TD会将通道数减半
                out_channels = current_channels // 2
                transition = TransitionDown(current_channels, out_channels, dropout_rate)
                self.transition_downs.append(transition)
                current_channels = out_channels

        # 瓶颈层（15层）
        self.bottleneck = DenseBlock(15, current_channels, growth_rate, dropout_rate)
        bottleneck_out_channels = current_channels + 15 * growth_rate

        # 解码器路径
        self.transition_ups = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        # 反向处理编码器配置
        for i, num_layers in enumerate(decoder_layers):
            # 过渡上采样
            in_ch = bottleneck_out_channels if i == 0 else current_channels
            tu_out_channels = growth_rate * 8
            transition_up = TransitionUp(in_ch, tu_out_channels)
            self.transition_ups.append(transition_up)

            # 解码器输入 = TU输出 + 跳跃连接
            # 跳跃连接来自编码器路径，通道数需要匹配
            skip_idx = len(encoder_layers) - 2 - i
            decoder_in_channels = tu_out_channels + skip_connections_channels[skip_idx]

            # 解码器密集块（输入不与输出拼接）
            # 论文中明确指出解码器是串行连接
            decoder_block = SerialDenseBlock(num_layers, decoder_in_channels, growth_rate, dropout_rate)
            self.decoder_blocks.append(decoder_block)

            current_channels = growth_rate * num_layers

        # 最终输出层：1x1卷积 + Softmax
        self.final_conv = nn.Conv2d(current_channels, num_classes, kernel_size=1)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 初始卷积
        x = self.initial_conv(x)
        skip_connections = [x]

        # 编码器前向传播
        for i in range(len(self.encoder_blocks) - 1):
            x = self.encoder_blocks[i](x)
            skip_connections.append(x)
            x = self.transition_downs[i](x)

        x = self.encoder_blocks[-1](x)
        skip_connections.append(x)
        x = self.bottleneck(x)

        # 解码器前向传播
        skip_idx = len(skip_connections) - 2

        for i in range(len(self.transition_ups)):
            # 上采样
            x = self.transition_ups[i](x)

            # 获取对应的跳跃连接
            skip = skip_connections[skip_idx]

            # 调整尺寸匹配
            if x.size()[-2:] != skip.size()[-2:]:
                x = F.interpolate(x, size=skip.size()[-2:], mode='bilinear', align_corners=True)

            # 与跳跃连接拼接
            x = torch.cat([x, skip], dim=1)

            # 解码器密集块（串行连接）
            x = self.decoder_blocks[i](x)
            skip_idx -= 1
        x = self.final_conv(x)
        return x