from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
from collections import OrderedDict

import _init_paths
from core.config import config

basic_block_num = 2
conv_per_block = 3
norm_type = 'instance_norm'  # batch_norm / l2_norm / instance_norm
in_channels = [64, 64]
out_channels = [64, 64]
kernel_size = [7, 7]
stride = [2, 2]
padding = [3, 3]



def flatten(x):
    N = x.shape[0]
    return x.view(N, -1)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return flatten(x)

class L2_Norm(nn.Module):
    def __init__(self):
        super(L2_Norm, self).__init__()

    def forward(self, x):
        print(x.norm(dim=(2, 3)).mean())
        return x / (x.norm(dim=(3), keepdim=True) + 1e-5)

# input.shape = [batch_size, channel_num, temporal_len, feature_dim]

class Basic_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Basic_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels[0], out_channels=out_channels[0],
                               kernel_size=(kernel_size[0], 1), stride=(stride[0], 1), padding=(padding[0], 0))
        # self.norm1 = nn.BatchNorm2d(num_features=out_channels[0])
        self.norm1 = nn.InstanceNorm2d(out_channels[0])
        # self.norm1 = L2_Norm()
        # self.norm1 = nn.Sequential(nn.InstanceNorm2d(out_channels[0]), L2_Norm())

        self.conv2 = nn.Conv2d(in_channels=in_channels[1], out_channels=out_channels[1],
                               kernel_size=(kernel_size[1], 1), stride=(stride[1], 1), padding=(padding[1], 0))
        # self.norm2 = nn.BatchNorm2d(num_features=out_channels[1])
        self.norm2 = nn.InstanceNorm2d(out_channels[1])
        # self.norm2 = L2_Norm()
        # self.norm2 = nn.Sequential(nn.InstanceNorm2d(out_channels[0]), L2_Norm())


    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.norm2(x)

        return x


class Time_Conv(nn.Module):
    def __init__(self):
        super(Time_Conv, self).__init__()

        # 128 -> 64
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=out_channels[0],
                               kernel_size=(kernel_size[0], 1), stride=(stride[0], 1), padding=(padding[0], 0))
        # self.norm1 = nn.BatchNorm2d(num_features=out_channels[0])
        # self.norm1 = L2_Norm()
        self.norm1 = nn.InstanceNorm2d(out_channels[0])
        # self.norm1 = nn.Sequential(nn.InstanceNorm2d(out_channels[0]), L2_Norm())

        # 64 -> 32
        self.conv2 = nn.Conv2d(in_channels=in_channels[1], out_channels=out_channels[1],
                               kernel_size=(kernel_size[1], 1), stride=(stride[1], 1), padding=(padding[1], 0))
        # self.norm2 = nn.BatchNorm2d(num_features=out_channels[1])
        # self.norm2 = L2_Norm()
        self.norm2 = nn.InstanceNorm2d(out_channels[1])
        # self.norm2 = nn.Sequential(nn.InstanceNorm2d(out_channels[0]), L2_Norm())

        # 32 -> 16
        self.relu1 = nn.ReLU()
        self.maxpooling1 = nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))

        # 16 -> 4
        self.layer1 = Basic_Block(in_channels, [out_channels[0], 1], kernel_size, stride, padding)

        # 4 -> 1
        self.relu2 = nn.ReLU()
        self.maxpooling2 = nn.MaxPool2d(kernel_size=(4, 1), stride=(1, 1), padding=(0, 0))

        # N * 1024
        self.final_layer = Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu1(x)
        x = self.maxpooling1(x)
        x = self.layer1(x)
        x = self.relu2(x)
        x = self.maxpooling2(x)
        x = self.final_layer(x)

        return x

def get_time_conv():
    return Time_Conv()
