''' Implementation of nonlocal module used in CCF-Net.
'''

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from ..plugins import CrissCrossAttention
from ..plugins import AFNB
from mmdet.models.plugins import GeneralizedAttention, NonLocal2D

class CAM_Module(nn.Module):
    """ Channel attention module """
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x, ref=None):
        """
            inputs :
                x : input feature maps( B X C X H X W )
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        if type(x) == type(ref):
            proj_query = x.view(m_batchsize, C, -1)
        else:
            proj_query = ref.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CFAMBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CFAMBlock, self).__init__()
        # origin impl is 1024. Noted by lzh
        #inter_channels = 1024
        inter_channels = in_channels
        self.conv_bn_relu1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())
        self.conv_bn_relu2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())

        #self.sc = CAM_Module(inter_channels)
        #self.sc = CrissCrossAttention(inter_channels)
        #self.sc = NonLocal2D(inter_channels, use_scale=False)
        self.sc = AFNB(inter_channels,
                inter_channels,
                inter_channels,
                inter_channels,
                inter_channels,
                dropout=0.00, norm_type=None)

        self.conv_bn_relu3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU())

        self.conv_out = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x, ref=None, loss=[]):

        x = self.conv_bn_relu1(x)
        x = self.conv_bn_relu2(x)
        #x, attention = self.sc(x, ref)
        #x = self.sc(x, ref)
        #print(x.shape, self.conv_bn_relu2,self.sc)
        x = self.sc(ref, x) # for afnb
        x = self.conv_bn_relu3(x)
        output = self.conv_out(x)

        return output


if __name__ == "__main__":
    data = torch.randn(18, 2473, 7, 7).cuda()
    in_channels = data.size()[1]
    out_channels = 145
    model = CFAMBlock(in_channels, out_channels).cuda()
    print(model)
    output = model(data)
    print(output.size())
