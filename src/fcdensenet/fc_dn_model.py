import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.putil import init_weights as init


class SliceAttentionExpand(nn.Module):
    def __init__(self, num_slices, rate=4):
        super(SliceAttentionExpand, self).__init__()
        self.num_slices = num_slices
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_slices, num_slices * rate),
            nn.ReLU(inplace=True),
            nn.Linear(num_slices * rate, num_slices),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        avg_pool = self.global_avg_pool(x).view(b, c)
        max_pool = self.global_max_pool(x).view(b, c)
        att = max_pool + avg_pool

        weights = self.fc(att).view(b, c, 1, 1)
        return x * weights

class FirstConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FirstConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout2d(0.2)
        self.bn = nn.InstanceNorm2d(in_channels)

    def forward(self, X):
        X = self.bn(X)
        X = F.relu(X)
        X = self.conv(X)
        X = self.dropout(X)
        return X


class DenseLayer(nn.Module):

    def __init__(self, input_channels, growth_rate, dropout_rate=0.2):
        super(DenseLayer, self).__init__()
        # BN + ReLU + 3x3 Conv + Dropout - 论文表1中的Layer结构
        # self.norm = nn.BatchNorm2d(input_channels)
        # IN + ReLU + 3x3 Conv + Dropout - 结合注意力模式
        self.norm = nn.InstanceNorm2d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(input_channels, growth_rate, kernel_size=3,
                              padding=1, bias=False)
        self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, X):
        X = self.norm(X)
        X = self.relu(X)
        X = self.conv(X)
        X = self.dropout(X)
        return X


class DenseBlock(nn.Module):

    def __init__(self, input_channels, num_layers, growth_rate, dropout_rate=0.2):
        super(DenseBlock, self).__init__()
        self.input_channels = input_channels
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            layer_input_channels = input_channels + i * growth_rate
            layer = DenseLayer(layer_input_channels, growth_rate, dropout_rate)
            self.layers.append(layer)

    def forward(self, X):
        features = [X]

        for layer in self.layers:
            input = torch.cat(features, dim=1)
            features.append(layer(input))

        return torch.cat(features, dim=1)

    def down_out_channels(self):
        return self.input_channels + self.num_layers * self.growth_rate

    def up_out_channels(self):
        return self.num_layers * self.growth_rate


class TransitionDown(nn.Module):
    def __init__(self, input_channels, dropout_rate=0.2):
        super(TransitionDown, self).__init__()
        # self.norm = nn.BatchNorm2d(input_channels)
        self.norm = nn.InstanceNorm2d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(input_channels, input_channels, kernel_size=1, bias=False)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, X):
        X = self.norm(X)
        X = self.relu(X)
        X = self.conv(X)
        X = self.dropout(X)
        X = self.pool(X)
        return X


class TransitionUp(nn.Module):
    def __init__(self, input_channels):
        super(TransitionUp, self).__init__()
        self.transpose_conv = nn.ConvTranspose2d(
            input_channels, input_channels,
            kernel_size=3, stride=2,
            padding=1, output_padding=1
        )

    def forward(self, X):
        return self.transpose_conv(X)

class FCDenseNet(nn.Module):
    def __init__(self, in_channels, first_channels, num_layers, growth_rate=18, dropout_rate=0.2):
        super(FCDenseNet, self).__init__()
        self.db_downs = nn.ModuleList()
        self.tds = nn.ModuleList()
        self.db_ups = nn.ModuleList()
        self.tus = nn.ModuleList()
        self.first_slice_attention = SliceAttentionExpand(3, 2)
        self.fist_conv = FirstConv(in_channels=in_channels, out_channels=first_channels)
        current_channels = first_channels

        layers = num_layers[:-1]
        skip_channels = []
        for layer in layers:
            self.db_downs.append(DenseBlock(current_channels, layer, growth_rate, dropout_rate=dropout_rate))
            current_channels = current_channels + layer * growth_rate
            self.tds.append(TransitionDown(current_channels, dropout_rate))
            skip_channels.append(current_channels)

        skip_channels.reverse()
        self.bottleneck_slice_attention = SliceAttentionExpand(current_channels, 2)
        self.bottleneck = DenseBlock(current_channels, num_layers[-1], growth_rate, dropout_rate)

        current_channels = num_layers[-1] * growth_rate
        layers.reverse()
        for i, layer in enumerate(layers):
            self.tus.append(TransitionUp(current_channels))
            current_channels = current_channels + skip_channels[i]
            self.db_ups.append(DenseBlock(current_channels, layer, growth_rate, dropout_rate))
            if i != len(num_layers) - 2:
                current_channels = layer * growth_rate
            else:
                current_channels = current_channels + layer * growth_rate

        self.output_conv = nn.Conv2d(in_channels=current_channels, out_channels=1, kernel_size=1)

    def forward(self, X):
        skip_x = []
        X = self.first_slice_attention(X)
        X = self.fist_conv(X)

        for i in range(len(self.db_downs)):
            X = self.db_downs[i](X)
            skip_x.append(X)
            X = self.tds[i](X)

        skip_x.reverse()
        X = self.bottleneck_slice_attention(X)
        X = self.bottleneck(X)
        last_bd = self.bottleneck
        for i in range(len(self.db_ups)):
            X = X[:, last_bd.input_channels:, ...]
            X = self.tus[i](X)
            X = torch.cat((skip_x[i], X), dim=1)
            X = self.db_ups[i](X)
            last_bd = self.db_ups[i]

        X = self.output_conv(X)

        return X
