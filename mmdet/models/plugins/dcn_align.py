import torch
from torch.nn import functional as F
import torch.nn as nn
import cv2
import os
import numpy as np
from mmdet.ops import DeformConv, ModulatedDeformConv
from mmcv.cnn import constant_init, normal_init, kaiming_init
#from ..utils import build_conv_layer, build_norm_layer

class Align_3D(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(Align_3D, self).__init__()
        self.aligner = Align_2D(in_channels)

    def forward(self, f_3d, f_2d=None):
        ''' f_2d:   2d feature which is usually the center feature (on depth-dim) of 3D feature map.
            f_3d:   3d feature.
        '''
        n, c, d, h, w = f_3d.shape
        center = d//2
        #if f_2d == None:
            #center_feature = f_3d[:, :, center, :, :]
        #center_feature = f_3d[:, :, center, :, :]
        center_feature = f_2d
        aligned_feature = []
        for i in range(d):
            aligned_feature.append(self.aligner(center_feature, f_3d[:, :, i, :, :]).unsqueeze(2))
        aligned_feature = torch.cat(aligned_feature, dim=2)
        return aligned_feature


class Align_2D(nn.Module):
    ''' Added by lzh.
        An feature alignment module implemented by Deformable Conv.
        Given two input features x and ref, align ref to x:
            offset = Conv(concat(x, ref))
            aligned_ref = DCN(ref, offset)
    '''
    def __init__(self, in_channels,
            dcn=dict(modulated=True, deformable_groups=1),
            **kwargs):

        super(Align_2D, self).__init__()
        self.in_channels = in_channels
        self.with_modulated_dcn = dcn.get('modulated', False)
        self.deformable_groups = dcn.get('deformable_groups', 1)
        self.offset_channels = 27 if self.with_modulated_dcn else 18
        self.conv_op = ModulatedDeformConv if self.with_modulated_dcn else DeformConv
        self.conv_cfg = dict(type='Conv')

        self.conv1 = self.conv_bn_relu(2 * self.in_channels, self.in_channels,
                kernel_size=1, padding=0)

        self.conv2_offset = nn.Conv2d(
            self.in_channels,
            self.deformable_groups * self.offset_channels,
            kernel_size=3,
            stride=1,
            padding=1)

        self.conv2 = self.conv_op(
            self.in_channels,
            self.in_channels,
            kernel_size=3,
            stride=1,
            deformable_groups=self.deformable_groups,
            padding=1,
            bias=False)
        self.norm2 = nn.Sequential(nn.BatchNorm2d(self.in_channels), nn.ReLU())
        self.conv3 = self.conv_bn_relu(self.in_channels, self.in_channels, 1, 0)
        self.conv4 = self.conv_bn_relu(self.in_channels, self.in_channels, 3, 1)

        self.init_weights()

    def conv_bn_relu(self, in_channels, out_channels, kernel_size=1, padding=0, bias=False):
        return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                    stride=1, padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.ReLU())

    def init_weights(self, std=0.01, zeros_init=True):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                constant_init(m, 1)
        constant_init(self.conv2_offset, 0)

    def forward(self, input, ref):
        out = self.conv1(torch.cat((input, ref),dim=1))
        if self.with_modulated_dcn:
            offset_mask = self.conv2_offset(out)
            offset = offset_mask[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv2(out, offset, mask)
        else:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)
        out = self.norm2(out)
        out = self.conv3(out)
        out = self.conv4(input + out)
        return out

if __name__ == '__main__':
    # DCN_Align
    #x = torch.rand(1,2,4,4).cuda()
    #ref = torch.rand(1,2,4,4).cuda()
    #dcn_align = Align_2D(2).cuda()
    #y = dcn_align(x, ref)
    #print(x.shape, y.shape)

    # 3D_Align
    x = torch.rand(1,2,4,4).cuda()
    ref = torch.rand(1,2,9,4,4).cuda()
    sa = Align_3D(2).cuda()
    y = sa(ref, x)
    print(x.shape, y.shape)


