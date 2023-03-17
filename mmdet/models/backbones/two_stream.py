import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.models.plugins import GeneralizedAttention
from mmdet.ops import ContextBlock, DeformConv, ModulatedDeformConv
from ..registry import BACKBONES
from ..utils import build_conv_layer, build_norm_layer

from .resnet_3D import ResNet_3D
from .resnet import ResNet, make_res_layer
from .fusion import CFAMBlock
import pdb

RESNET_OUT_CHANNELS = [256, 512,1024, 2048]

@BACKBONES.register_module
class TwoStreamResNet(nn.Module):
    def __init__(self, res2d, res3d, fusion_mode='conv',
            data_mode='three_slices_as_rgb'):
        super(TwoStreamResNet, self).__init__()
        self.res2d = ResNet(**res2d)
        self.res3d = ResNet_3D(**res3d)

        self.fusion_mode = fusion_mode
        self.data_mode = data_mode
        #self.fusion_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True) \
                #if 'gn' in self.fusion_mode else None
        if self.fusion_mode in ['channel_attention_cross', 'channel_attention',
                'channel_attention_weighted', 'conv', 'nonlocal', 'efficientnl',
                'nonlocal_woconv']:
            fusion_layers= nn.ModuleList()
            for i in range(len(self.res3d.res_layers)):
                fusion_layers.append(nn.Sequential(nn.Conv3d(RESNET_OUT_CHANNELS[i], RESNET_OUT_CHANNELS[i],
                    kernel_size=(7,1,1), bias=False),
                    nn.GroupNorm(num_groups=32, num_channels=RESNET_OUT_CHANNELS[i]),
                    nn.ReLU()))
            self.fusion_layers = nn.Sequential(*fusion_layers)

            if self.fusion_mode == 'channel_attention_weighted':
                pre_2d_layers= nn.ModuleList()
                for i in range(len(self.res3d.res_layers)):
                    #pre_2d_layers.append(build_conv_layer(
                        #dict(type='Conv'), RESNET_OUT_CHANNELS[i], RESNET_OUT_CHANNELS[i],
                        #kernel_size= 1))
                    pre_2d_layers.append(nn.Sequential(nn.Conv2d(RESNET_OUT_CHANNELS[i], RESNET_OUT_CHANNELS[i],
                        kernel_size=1, bias=False),
                        nn.GroupNorm(num_groups=32, num_channels=RESNET_OUT_CHANNELS[i]),
                        nn.ReLU()))
                self.pre_2d_layers = nn.Sequential(*pre_2d_layers)

                pre_3d_layers= nn.ModuleList()
                for i in range(len(self.res3d.res_layers)):
                    pre_3d_layers.append(nn.Sequential(nn.Conv3d(RESNET_OUT_CHANNELS[i], RESNET_OUT_CHANNELS[i],
                        kernel_size=(1,1,1), bias=False),
                        nn.GroupNorm(num_groups=32, num_channels=RESNET_OUT_CHANNELS[i]),
                        nn.ReLU()))
                    #pre_3d_layers.append(build_conv_layer(
                        #dict(type='Conv3d'), RESNET_OUT_CHANNELS[i], RESNET_OUT_CHANNELS[i],
                        #kernel_size = (1, 1, 1)))
                        #norm_cfg=self.fusion_norm_cfg)
                self.pre_3d_layers = nn.Sequential(*pre_3d_layers)
            elif self.fusion_mode in ['nonlocal', 'nonlocal_woconv']:
                nl_layers= nn.ModuleList()
                for i in range(len(self.res3d.res_layers)):
                    nl_layers.append(CFAMBlock(
                        RESNET_OUT_CHANNELS[i] * 2, RESNET_OUT_CHANNELS[i]))
                self.nl_layers = nn.Sequential(*nl_layers)
            elif self.fusion_mode=='efficientnl':
                self.avg2d = nn.AdaptiveAvgPool2d((8,8))
                self.avg3d = nn.AdaptiveAvgPool3d((8,8,8))
                nl_layers= nn.ModuleList()
                for i in range(len(self.res3d.res_layers)):
                    nl_layers.append(CFAMBlock(
                        RESNET_OUT_CHANNELS[i] * 2, RESNET_OUT_CHANNELS[i]))
                self.nl_layers = nn.Sequential(*nl_layers)
        elif self.fusion_mode in ['concat', 'simple_concat']:
            fusion_layers= nn.ModuleList()
            for i in range(len(self.res3d.res_layers)):
                fusion_layers.append(build_conv_layer(
                    dict(type='Conv'), 2 * RESNET_OUT_CHANNELS[i], RESNET_OUT_CHANNELS[i],
                    kernel_size=1))
            self.fusion_layers = nn.Sequential(*fusion_layers)


    def forward(self, x):
        # input x : n,c,d,h,w
        #out_2d = self.res2d(x.squeeze(1))
        #out_3d = self.res3d(x)
        n, c, d, h, w = x.shape
        if self.data_mode == 'three_slices_as_rgb':
            x1 = x[:, 0, d//2-1 : d//2+2, :, :]
        elif self.data_mode in ['single_slice_as_rgb', 'single_slice']:
            x1 = x[:, 0, (d//2, d//2, d//2), :, :]
        # if set 'single_slice', in_channels of Res2D should set 1.
        #elif self.data_mode == 'single_slice':
            #x1 = x[:, 0, d//2:d//2+1, :, :]
        x1 = self.res2d.conv1(x1)
        x1 = self.res2d.norm1(x1)
        x1 = self.res2d.relu(x1)
        x1 = self.res2d.maxpool(x1)

        x2 = self.res3d.conv1(x)
        x2 = self.res3d.norm1(x2)
        x2 = self.res3d.relu(x2)
        x2 = self.res3d.maxpool(x2)

        outs = []
        for i in range(len(self.res2d.res_layers)):
            res_layer_2d = getattr(self.res2d, self.res2d.res_layers[i])
            res_layer_3d = getattr(self.res3d, self.res3d.res_layers[i])
            x1 = res_layer_2d(x1)
            x2 = res_layer_3d(x2)
            if self.fusion_mode in ['nonlocal', 'nonlocal_woconv']:
                x1, x2 = self.fusion(x1, x2, i)
                if i in self.res2d.out_indices:
                    outs.append(x1)
            elif self.fusion_mode == 'simple_concat':
                n, c, d, h, w = x2.shape
                concated_feat = torch.cat((x1, x2[:, :, d//2, :, :]), dim=1)
                x1 = self.fusion_layers[i](concated_feat)
                outs.appent(x1)
            else:
                x1 = self.fusion(x1, x2, i)
                if i in self.res2d.out_indices:
                    outs.append(x1)
        return tuple(outs)

    def fusion(self, x1, x2, idx):
        if self.fusion_mode == 'conv':
            x2 = self.fusion_layers[idx](x2)
            n, c, d, h, w = x2.shape
            assert d==1
            x2 = x2[:, :, 0, :, :]
            out = x1 + x2
            return out
        elif self.fusion_mode == 'concat':
            n, c, d, h, w = x2.shape
            concated_feat = torch.cat((x1, x2[:, :, d//2, :, :]), dim=1)
            concat_out = self.fusion_layers[idx](concated_feat)
            return concat_out
        # TODO: will bn or relu help?
        elif self.fusion_mode == 'channel_attention_cross':
            # this is a kind of position-aware attention between 2D and 3D features.
            n, c, d, h, w = x2.shape
            simi_list = []
            for i in range(d):
                simi_list.append(torch.sum(x1 * x2[:, :, i, :, :], 1).unsqueeze(1))
                #simi_list.append(torch.sum(x2_center * x2[:, :, i, :, :], 1).unsqueeze(1))
            # [n, 1, h, w]*num_slice --> n, d, h, w -->  n, c, d, h, w
            similarity = torch.sigmoid(torch.cat(simi_list, dim=1)).unsqueeze(1).repeat(1, c, 1, 1, 1)
            x2 = x2 * similarity
            # same as 'conv'
            x2 = self.fusion_layers[idx](x2)
            n, c, d, h, w = x2.shape
            assert d==1
            x2 = x2[:, :, 0, :, :]
            out = x1 + x2
            return out
        elif self.fusion_mode == 'channel_attention':
            # this is a kind of position-aware attention between 2D and 3D features.
            n, c, d, h, w = x2.shape
            simi_list = []
            x2_center = x2[:, :, d//2, :, :]
            for i in range(d):
                #simi_list.append(torch.sum(x1 * x2[:, :, i, :, :], 1).unsqueeze(1))
                simi_list.append(torch.sum(x2_center * x2[:, :, i, :, :], 1).unsqueeze(1))
            # [n, 1, h, w]*num_slice --> n, d, h, w -->  n, c, d, h, w
            similarity = torch.sigmoid(torch.cat(simi_list, dim=1)).unsqueeze(1).repeat(1, c, 1, 1, 1)
            x2 = x2 * similarity
            # same as 'conv'
            x2 = self.fusion_layers[idx](x2)
            n, c, d, h, w = x2.shape
            assert d==1
            x2 = x2[:, :, 0, :, :]
            out = x1 + x2
            return out
        elif self.fusion_mode == 'channel_attention_weighted':
            # this is a kind of position-aware attention between 2D and 3D features.
            n, c, d, h, w = x2.shape
            simi_list = []
            x1 = self.pre_2d_layers[idx](x1)
            x2 = self.pre_3d_layers[idx](x2)
            x2_center = x2[:, :, d//2, :, :]
            for i in range(d):
                #simi_list.append(torch.sum(x1 * x2[:, :, i, :, :], 1).unsqueeze(1))
                simi_list.append(torch.sum(x2_center * x2[:, :, i, :, :], 1).unsqueeze(1))
            # [n, d, h, w]*num_slice --> n, 1, d, h, w -->  n, c, d, h, w
            similarity = torch.sigmoid(torch.cat(simi_list, dim=1)).unsqueeze(1).repeat(1, c, 1, 1, 1)
            x2 = x2 * similarity
            # same as 'conv'
            x2 = self.fusion_layers[idx](x2)
            n, c, d, h, w = x2.shape
            assert d==1
            x2 = x2[:, :, 0, :, :]
            out = x1 + x2
            return out
        elif self.fusion_mode == 'nonlocal':
            # this is a kind of position-agnostic attention between 2D and 3D features.
            n, c, d, h, w = x2.shape
            nl_feats = []
            for i in range(d):
                feat = self.nl_layers[idx](torch.cat((x1, x2[:, :, i, :, :]), dim=1)).unsqueeze(2)
                nl_feats.append(feat)
            nl_feats = torch.cat(nl_feats, dim=2)
            ##########
            attention = self.fusion_layers[idx](nl_feats)
            n, c, d, h, w = attention.shape
            assert d==1
            out_2d = x1 + attention[:, :, 0, :, :]
            ########## without 711 conv3d
            #n, c, d, h, w =  nl_feats.shape
            #out_2d = x1 + nl_feats[:, :, d//2, :, :]
            ##########
            out_3d = x2 + nl_feats
            return out_2d, out_3d
        elif self.fusion_mode == 'nonlocal_woconv':
            # this is a kind of position-agnostic attention between 2D and 3D features.
            n, c, d, h, w = x2.shape
            nl_feats = []
            for i in range(d):
                feat = self.nl_layers[idx](torch.cat((x1, x2[:, :, i, :, :]), dim=1)).unsqueeze(2)
                nl_feats.append(feat)
            nl_feats = torch.cat(nl_feats, dim=2)
            ##########
            #attention = self.fusion_layers[idx](nl_feats)
            #n, c, d, h, w = attention.shape
            #assert d==1
            #out_2d = x1 + attention[:, :, 0, :, :]
            ########## without 711 conv3d
            n, c, d, h, w =  nl_feats.shape
            out_2d = x1 + nl_feats[:, :, d//2, :, :]
            ##########
            out_3d = x2 + nl_feats
            return out_2d, out_3d
        elif self.fusion_mode == 'efficientnl':
            n, c, d, h, w = x2.shape
            # (n,c)
            x1_global = self.avg2d(x1)
            # (n,c,d)
            x2_global = self.avg3d(x2)
            nl_feats = []
            for i in range(d):
                feat = self.nl_layers[idx](torch.cat((x1_global, x2_global[:, :, i, :, :]), dim=1)).unsqueeze(2)
                nl_feats.append(feat)
                #proj_query = x.view(m_batchsize, C, -1)
                #proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
                #energy = torch.bmm(proj_query, proj_key)
                #energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
                #attention = self.softmax(energy_new)
                #proj_value = x.view(m_batchsize, C, -1)

                #out = torch.bmm(attention, proj_value)
                #out = out.view(m_batchsize, C, height, width)
            out_3d = torch.cat(nl_feats, dim=2).repeat(1, 1, 1, h, w)
            x2 = self.fusion_layers[idx](out_3d)
            n, c, d, h, w = x2.shape
            assert d==1
            out_2d = x1 + x2[:, :, 0, :, :]
            return out_2d, out_3d
        else:
            raise NotImplementedError("this funcion is not implemented!")


    def init_weights(self, pretrained=None):
        if pretrained == None:
            pretrained = [None, None]
        self.res2d.init_weights(pretrained[0])
        self.res3d.init_weights(pretrained[1])

    def train(self, mode=True):
        self.res2d.train(mode)
        self.res3d.train(mode)

