'''
Description: 
Author: Xiongjun Guan
Date: 2024-05-22 15:54:18
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2024-05-22 16:00:44

Copyright (C) 2024 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class h_sigmoid(nn.Module):

    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):

    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):

    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class ASPP(nn.Module):

    def __init__(self, in_channel, depth, dilations=[1, 2, 4, 8]):
        super(ASPP, self).__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, depth, kernel_size=1, stride=1),
            nn.BatchNorm2d(depth),
            nn.ReLU(inplace=True),
        )
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Sequential(
            nn.Conv2d(in_channel, depth, kernel_size=1, stride=1),
            nn.BatchNorm2d(depth),
            nn.ReLU(inplace=True),
        )
        self.atrous_block2 = nn.Sequential(
            nn.Conv2d(in_channel,
                      depth,
                      kernel_size=3,
                      stride=1,
                      padding=dilations[1],
                      dilation=dilations[1]),
            nn.BatchNorm2d(depth),
            nn.ReLU(inplace=True),
        )
        self.atrous_block3 = nn.Sequential(
            nn.Conv2d(in_channel,
                      depth,
                      kernel_size=3,
                      stride=1,
                      padding=dilations[2],
                      dilation=dilations[2]),
            nn.BatchNorm2d(depth),
            nn.ReLU(inplace=True),
        )
        if len(dilations) == 3:
            self.atrous_block4 = None
        else:
            self.atrous_block4 = nn.Sequential(
                nn.Conv2d(in_channel,
                          depth,
                          kernel_size=3,
                          stride=1,
                          padding=dilations[3],
                          dilation=dilations[3]),
                nn.BatchNorm2d(depth),
                nn.ReLU(inplace=True),
            )

        if len(dilations) == 3:
            self.conv_1x1_output = nn.Sequential(
                nn.Conv2d(depth * 4, depth, kernel_size=1, stride=1),
                nn.BatchNorm2d(depth),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv_1x1_output = nn.Sequential(
                nn.Conv2d(depth * 5, depth, kernel_size=1, stride=1),
                nn.BatchNorm2d(depth),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.interpolate(image_features,
                                       size=size,
                                       mode="bilinear",
                                       align_corners=True)

        atrous_block1 = self.atrous_block1(x)
        atrous_block2 = self.atrous_block2(x)
        atrous_block3 = self.atrous_block3(x)
        if self.atrous_block4 is not None:
            atrous_block4 = self.atrous_block4(x)
            out = self.conv_1x1_output(
                torch.cat([
                    image_features, atrous_block1, atrous_block2,
                    atrous_block3, atrous_block4
                ],
                          dim=1))
        else:
            out = self.conv_1x1_output(
                torch.cat([
                    image_features, atrous_block1, atrous_block2, atrous_block3
                ],
                          dim=1))
        return out


class DownBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=kernel_size,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,
                      out_channels,
                      kernel_size=kernel_size,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.conv(x)


class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=kernel_size,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class OutConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1), )

    def forward(self, x):
        return self.conv(x)


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     padding=1,
                                     stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride
        if in_channels != out_channels:
            self.in_conv = torch.nn.Conv2d(in_channels,
                                           out_channels,
                                           kernel_size=1,
                                           stride=stride)
        else:
            self.in_conv = None

    def forward(self, x):
        if self.in_conv is None:
            identity = x
        else:
            identity = self.in_conv(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class FeatureDownSampleBlock(nn.Module):

    def __init__(self):
        super(FeatureDownSampleBlock, self).__init__()
        self.down1 = DownBlock(1, 32)
        self.down2 = DownBlock(32, 64)
        self.down3 = DownBlock(64, 128)
        self.down4 = DownBlock(128, 256)

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        return x


class FeatBlock(nn.Module):

    def __init__(self, in_channels=256, out_channels=256):
        super(FeatBlock, self).__init__()
        self.conv1 = ConvLayer(in_channels, out_channels)
        self.conv2 = ConvLayer(out_channels, out_channels)
        self.conv3 = ConvLayer(out_channels, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class DDRNet_DIR(nn.Module):

    def __init__(self, dis_const=16):
        super(DDRNet_DIR, self).__init__()
        self.dis_const = dis_const

        self.down_block = FeatureDownSampleBlock()

        self.dir_block = FeatBlock(256, 256)
        self.dir_output = nn.Sequential(
            nn.Conv2d(256, 180, kernel_size=1, stride=1, padding=0),
            nn.Softmax2d(),
        )

        self.dir_res1 = ResBlock(256, 256)
        self.dir_res2 = ResBlock(256, 256)
        self.dir_att = CoordAtt(256, 256, reduction=16)

        self.res1 = ResBlock(256, 256)
        self.res2 = ResBlock(256, 256)
        self.img_att = CoordAtt(256, 256, reduction=16)

        self.res_bone1 = ResBlock(513, 513)
        self.res_bone2 = ResBlock(513, 513)
        self.bone_att = CoordAtt(513, 513, reduction=16)

        self.aspp = ASPP(513, 1024)
        self.aspp_att = CoordAtt(1024, 1024, reduction=16)

        self.end_conv1 = ConvLayer(1024, 512)
        self.end_conv2 = ConvLayer(512, 256)
        self.end_conv3 = ConvLayer(256, 256)
        self.end_conv4 = ConvLayer(256, 256)
        self.conv_out = OutConv(256, 2)

    def forward(self, x, mask):
        x = self.down_block(x)

        dir_feat = self.dir_block(x)
        dir_out = self.dir_output(dir_feat)

        dir_feat = self.dir_res1(dir_feat)
        dir_feat = self.dir_res2(dir_feat)
        dir_feat = self.dir_att(dir_feat)

        img_feat = self.res1(x)
        img_feat = self.res2(img_feat)
        img_feat = self.img_att(img_feat)

        x = torch.cat([img_feat, dir_feat, mask], 1)
        x = self.res_bone1(x)
        x = self.res_bone2(x)
        x = self.bone_att(x)

        x = self.aspp(x)
        x = self.aspp_att(x)

        x = self.end_conv1(x)
        x = self.end_conv2(x)
        x = self.end_conv3(x)
        x = self.end_conv4(x)
        x = self.conv_out(x) * self.dis_const

        return x, dir_out
