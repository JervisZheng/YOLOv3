import torch
from torch import nn
from common import *


class DarkNet53(nn.Module):
    def __init__(self, in_channels, num_class):
        super().__init__()
        self.num_class = num_class
        self.backbone1 = nn.Sequential(
            Conv(in_channels, 32, 3, 1, 1),
            Conv(32, 64, 3, 2, 1),

            ResConvBlock(64, 128, 3, 2),

            ResConvBlock(128, 128, 3),
            ResConvBlock(128, 256, 3, 2),

            ResConvBlock(256, 256, 3),
            ResConvBlock(256, 256, 3),
            ResConvBlock(256, 256, 3),
            ResConvBlock(256, 256, 3),
            ResConvBlock(256, 256, 3),
            ResConvBlock(256, 256, 3),
            ResConvBlock(256, 256, 3),
            ResConvBlock(256, 256, 3)
        )

        self.backbone2 = nn.Sequential(
            ResConvBlock(256, 512, 3, 2),
            ResConvBlock(512, 512, 3),
            ResConvBlock(512, 512, 3),
            ResConvBlock(512, 512, 3),
            ResConvBlock(512, 512, 3),
            ResConvBlock(512, 512, 3),
            ResConvBlock(512, 512, 3),
            ResConvBlock(512, 512, 3),
        )

        self.backbone3 = nn.Sequential(
            ResConvBlock(512, 1024, 3, 2),
            ResConvBlock(1024, 1024, 3),
            ResConvBlock(1024, 1024, 3),
            ResConvBlock(1024, 1024, 3)
        )

        self.up = nn.Upsample(scale_factor=2)

        self.convset1 = ConvSet(1024)
        self.convset2 = ConvSet(1024)
        self.convset3 = ConvSet(768)

        self.head1 = DetectHead(512, (self.num_class+5)*3)
        self.head2 = DetectHead(512, (self.num_class+5)*3)
        self.head3 = DetectHead(384, (self.num_class+5)*3)

    def forward(self, x):
        b1 = self.backbone1(x)
        b2 = self.backbone2(b1)
        b3 = self.backbone3(b2)

        # 侦测头1
        out1_ = self.convset1(b3)
        out1 = self.head1(out1_)

        # 侦测头2
        out2_ = self.up(out1_)
        out2_ = torch.cat([out2_, b2], dim=1)
        out2_ = self.convset2(out2_)
        out2 = self.head2(out2_)

        # 侦测头3
        out3_ = self.up(out2_)
        out3_ = torch.cat([out3_, b1], dim=1)
        out3_ = self.convset3(out3_)
        out3 = self.head3(out3_)
        return out1, out2, out3


if __name__ == '__main__':
    data = torch.randn(1, 3, 608, 608)
    model = DarkNet53(3, 10)
    out = model(data)
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)
    # print(model)
