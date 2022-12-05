# -*- coding: UTF-8 -*-
# @Time: 2022.12.05
# @Author: Jervis

import torch
from torch import nn


class Conv(nn.Module):
    """ConvUnit"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 use_bn=True,
                 act=nn.ReLU
                 ):
        super().__init__()

        if use_bn:
            bias = False
        else:
            bias = True

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias),
            nn.BatchNorm2d(out_channels) if use_bn else nn.Sequential(),
            act(inplace=True) if isinstance(act, nn.Module) else nn.Sequential()
        )

    def forward(self, x):
        return self.layer(x)


class ResConvBlock(nn.Module):
    """残差卷积模块， 瓶颈结"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 ratio=2):
        super().__init__()
        assert stride in [1, 2]
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        hidden_channels = in_channels // ratio

        self.layer = nn.Sequential(
            Conv(in_channels, hidden_channels, kernel_size=1, stride=1, act=nn.LeakyReLU),
            Conv(hidden_channels, out_channels, kernel_size=3, stride=stride, padding=kernel_size // 2,
                 act=nn.LeakyReLU)
        )

    def forward(self, x):
        if self.stride == 1 and self.in_channels == self.out_channels:
            return self.layer(x) + x
        else:
            return self.layer(x)


class DetectHead(nn.Module):
    """侦测头模块"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 ratio=2):
        super().__init__()
        hidden_channels = in_channels * ratio
        self.head = nn.Sequential(
            Conv(in_channels, hidden_channels, kernel_size, stride, padding=kernel_size // 2),
            Conv(hidden_channels, out_channels, kernel_size=1, stride=1, use_bn=False, act=False)
        )

    def forward(self, x):
        return self.head(x)


class ConvSet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.convset = nn.Sequential(
            Conv(in_channels, in_channels // 2, 1, 1),
            Conv(in_channels // 2, in_channels, 3, 1, 1),
            Conv(in_channels, in_channels // 2, 1, 1),
            Conv(in_channels // 2, in_channels, 3, 1, 1),
            Conv(in_channels, in_channels // 2, 1, 1)
        )

    def forward(self, x):
        return self.convset(x)


if __name__ == '__main__':
    data = torch.randn(1, 3, 640, 640)
    model = Conv(3, 64, 3, 1, 1)
    out = model(data)
    print(out.shape)
