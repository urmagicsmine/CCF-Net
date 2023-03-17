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

#from .resnet_3D import ResNet_3D
from .modified_p3d import modified_P3D
from .p3d import P3D
from .resnet import ResNet, make_res_layer, BasicBlock, Bottleneck
from .fusion import CFAMBlock
from .cross_layer_fusion import CSABlock
from ..plugins import CrissCrossAttention, CCAttention
from ..plugins import Align_2D, Align_3D
from ..plugins import AsyNonLocal2D
from ..plugins import AFNB
from ..plugins.cbam import *
import pdb


#RESNET_OUT_CHANNELS = [256, 512,1024, 2048]
RESNET_OUT_CHANNELS = [64, 128, 256, 512]

@BACKBONES.register_module
class TwoStreamP3d(nn.Module):
    def __init__(self, res2d, res3d, fusion_mode='conv',
            data_mode='three_slices_as_rgb', num_slices=7):
        super(TwoStreamP3d, self).__init__()
        self.res2d = ResNet(**res2d)
        self.res3d = P3D(**res3d)

        self.fusion_mode = fusion_mode
        self.data_mode = data_mode
        self.num_slices = num_slices
        self.num_depths = []
        #TODO: depth_current_stage should be re-computed.
        # Z-axis pooling is applied for p3d backbone. calculate num of depth channel of /
        # each stage. e.g. for num_slices=7, depth channel of res_layer_1 = (7//2)//2 = 1
        depth_current_stage = num_slices
        #for i in range(len(self.res2d.res_layers)):
            #depth_current_stage = depth_current_stage//2
            #self.num_depths.append(max(depth_current_stage,1))
            #print('Depth of resnet stage {}:', max(depth_current_stage,1))
        self.num_depths = [9,5,3,1]
        if self.fusion_mode in ['channel_attention_cross', 'channel_attention',
                'channel_attention_weighted', 'conv', 'nonlocal', 'efficientnl',
                'nonlocal_woconv', 'ccnet_nonlocal', 'concat_resblock']:
            fusion_layers= nn.ModuleList()
            for i in range(len(self.res3d.res_layers)):
                #fusion_layers.append(nn.Sequential(
                    #nn.Conv2d(RESNET_OUT_CHANNELS[i], RESNET_OUT_CHANNELS[i],
                    #kernel_size=1, bias=False),
                    #nn.GroupNorm(num_groups=32, num_channels=RESNET_OUT_CHANNELS[i]),
                    #nn.ReLU()))
                fusion_layers.append(nn.Sequential(
                    nn.Conv3d(RESNET_OUT_CHANNELS[i], RESNET_OUT_CHANNELS[i],
                    kernel_size=(self.num_depths[i],1,1), bias=False),
                    nn.GroupNorm(num_groups=32, num_channels=RESNET_OUT_CHANNELS[i]),
                    nn.ReLU()
                    ))
            self.fusion_layers = nn.Sequential(*fusion_layers)

            if self.fusion_mode == 'channel_attention_weighted':
                pre_2d_layers= nn.ModuleList()
                for i in range(len(self.res3d.res_layers)):
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
                self.pre_3d_layers = nn.Sequential(*pre_3d_layers)
            elif self.fusion_mode in ['nonlocal', 'nonlocal_woconv', 'ccnet_nonlocal']:
                if self.fusion_mode == 'ccnet_nonlocal':
                    nl_module=CCAttention
                else:
                    nl_module=CFAMBlock
                nl_layers= nn.ModuleList()
                align_layers = nn.ModuleList()
                se_layers = nn.ModuleList()
                for i in range(len(self.res3d.res_layers)):
                    nl_layers.append(nl_module(
                        RESNET_OUT_CHANNELS[i] * 2, RESNET_OUT_CHANNELS[i]))
                    #nl_layers.append(nl_module(
                        #RESNET_OUT_CHANNELS[i], RESNET_OUT_CHANNELS[i]))
                    #nl_layers.append(AsyNonLocal2D(RESNET_OUT_CHANNELS[i], RESNET_OUT_CHANNELS[i]))
                    #num_c = RESNET_OUT_CHANNELS[i]
                    #nl_layers.append((AFNB(num_c, num_c, num_c, num_c//2, num_c//2, dropout=0.05)))
                    #align_layers.append(Align_3D(RESNET_OUT_CHANNELS[i]))
                    #align_layers.append(Align_3D(RESNET_OUT_CHANNELS[i]))
                    #se_layers.append(CBAM(RESNET_OUT_CHANNELS[i], 16))
                self.nl_layers = nn.Sequential(*nl_layers)
                #self.se_layers = nn.Sequential(*se_layers)
                #self.align_layers = nn.Sequential(*align_layers)

        elif self.fusion_mode in ['concat_resblock']:
            fusion_layers= nn.ModuleList()
            for i in range(len(self.res3d.res_layers)):
                fusion_layers.append(
                    make_res_layer(
                        BasicBlock, RESNET_OUT_CHANNELS[i] * 2, RESNET_OUT_CHANNELS[i], 2,
                        norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
                        ))
            self.fusion_layers = nn.Sequential(*fusion_layers)

        elif self.fusion_mode in ['concat', 'simple_concat']:
            fusion_layers= nn.ModuleList()
            for i in range(len(self.res3d.res_layers)):
                fusion_layers.append(build_conv_layer(
                    dict(type='Conv'), 2 * RESNET_OUT_CHANNELS[i], RESNET_OUT_CHANNELS[i],
                    kernel_size=1))
            self.fusion_layers = nn.Sequential(*fusion_layers)


    def forward(self, x, losses=dict()):
        # input x : n,c,d,h,w
        #out_2d = self.res2d(x.squeeze(1))
        #out_3d = self.res3d(x)
        n, c, d, h, w = x.shape
        if self.data_mode == 'three_slices_as_rgb':
            x1 = x[:, 0, d//2-1 : d//2+2, :, :]
        elif self.data_mode in ['single_slice_as_rgb', 'single_slice']:
            x1 = x[:, 0, (d//2, d//2, d//2), :, :]
        x1 = self.res2d.conv1(x1)
        x1 = self.res2d.norm1(x1)
        x1 = self.res2d.relu(x1)
        x1 = self.res2d.maxpool(x1)

        x2 = self.res3d.conv1_custom(x)
        x2 = self.res3d.norm1(x2)
        x2 = self.res3d.relu(x2)
        x2 = self.res3d.maxpool(x2)

        outs = []
        entropy_loss = list()
        for i in range(len(self.res2d.res_layers)):
            res_layer_2d = getattr(self.res2d, self.res2d.res_layers[i])
            res_layer_3d = self.res3d.res_layers[i]
            x1 = res_layer_2d(x1)
            x2 = res_layer_3d(x2)
            if i > 2:
                x2 = self.res3d.pool_list[i](x2)
                outs.append(x1)
                continue
            if self.fusion_mode in ['nonlocal', 'nonlocal_woconv', 'ccnet_nonlocal']:
                #x1, x2 = self.fusion(x1, x2, i)
                x1, x2, out = self.fusion(x1, x2, i)
                if i in self.res2d.out_indices:
                    #outs.append(x1)
                    outs.append(out)
            elif self.fusion_mode == 'simple_concat':
                n, c, d, h, w = x2.shape
                concated_feat = torch.cat((x1, x2[:, :, d//2, :, :]), dim=1)
                x1 = self.fusion_layers[i](concated_feat)
                outs.append(x1)
                #x1_out = self.fusion_layers[i](concated_feat) # add extra 1x1ConvGnRelu is harmful(77->66)
                #x1_out = self.fusion_layers[i](concated_feat)
                #outs.append(x1_out)
            elif self.fusion_mode == 'simple_sum':
                n, c, d, h, w = x2.shape
                x1 = x1 + x2[:, :, d//2, :, :]
                #outs.append(x1+ x2[:, :, d//2, :, :])
                outs.append(x1)
            elif self.fusion_mode == 'concat_resblock':
                n, c, d, h, w = x2.shape
                concated_feat = torch.cat((x1, x2[:, :, d//2, :, :]), dim=1)
                x1 = self.fusion_layers[i](concated_feat)
                outs.append(x1)
            else:
                x1 = self.fusion(x1, x2, i)
                if i in self.res2d.out_indices:
                    outs.append(x1)
            x2 = self.res3d.pool_list[i](x2)
            #x2 = self.res3d.maxpool_2(x2)
        losses.update(dict(loss_MatrixEntropy=entropy_loss))
        #for idx, item in enumerate(outs):
            #print(idx, item.shape)
        return tuple(outs)

    def fusion(self, x1, x2, idx, loss=[]):
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
        elif self.fusion_mode == 'backup_nonlocal':
            # this is a kind of position-agnostic attention between 2D and 3D features.
            n, c, d, h, w = x2.shape
            #TODO this align position doesn't make sense
            #x2_ = self.align_layers[idx](x2, x1)
            nl_feats = []
            for i in range(d):
                feat = self.nl_layers[idx](torch.cat((x1, x2_[:, :, i, :, :]), dim=1)).unsqueeze(2)
                nl_feats.append(feat)
            nl_feats = torch.cat(nl_feats, dim=2)
            attention = self.fusion_layers[idx](nl_feats)
            n, c, d, h, w = attention.shape
            out_2d = x1 + attention[:, :, d//2, :, :]
            return out_2d, x2
            #out_3d = x2 + nl_feats
            #return out_2d, out_3d
        elif self.fusion_mode == 'nonlocal':
            # this is a kind of position-agnostic attention between 2D and 3D features.
            n, c, d, h, w = x2.shape
            x2_ref = x2.view(n, c, -1, w).contiguous()
            nl_feats = self.nl_layers[idx](x1, x2_ref)
            #out_2d = x1 + nl_feats
            out_2d = self.fusion_layers[idx](nl_feats)
            #attention = self.fusion_layers[idx](nl_feats)
            #n, c, d, h, w = attention.shape
            #out_2d = x1 + attention[:, :, d//2, :, :]
            return out_2d, x2
        elif self.fusion_mode == 'ccnet_nonlocal':
            # this is a kind of position-agnostic attention between 2D and 3D features.
            n, c, d, h, w = x2.shape
            nl_feats = []
            #center_feature = x2[:, :, d//2, :, :]
            for i in range(d):
                feat = self.nl_layers[idx](torch.cat((x1, x2[:, :, i, :, :]), dim=1), loss=loss).unsqueeze(2)
                #feat = self.nl_layers[idx](x2[:,:,i,:,:], center_feature).unsqueeze(2)
                nl_feats.append(feat)
            nl_feats = torch.cat(nl_feats, dim=2)
            #nl_feats = self.nl_layers[idx](x2, center_feature).unsqueeze(2)
            #x2_ref = x2.view(n, c, -1, w).contiguous()
            #nl_feats = self.nl_layers[idx](x1, center_feature)
            attention = self.fusion_layers[idx](nl_feats)
            n, c, d_, h, w = attention.shape
            #out_2d = x1 + attention[:, :, d_//2, :, :]
            out_2d = x1
            out = x2[:, :, d//2, :, :] + attention[:, :, d_//2, :, :]
            out_3d = x2
            return out_2d, out_3d, out
        elif self.fusion_mode == 'nonlocal_woconv':
            # this is a kind of position-agnostic attention between 2D and 3D features.
            n, c, d, h, w = x2.shape
            nl_feats = []
            for i in range(d):
                feat = self.nl_layers[idx](torch.cat((x1, x2[:, :, i, :, :]), dim=1)).unsqueeze(2)
                nl_feats.append(feat)
            nl_feats = torch.cat(nl_feats, dim=2)
            n, c, d, h, w =  nl_feats.shape
            out_2d = x1 + nl_feats[:, :, d//2, :, :]
            out_3d = x2 + nl_feats
            return out_2d, out_3d
        elif 'csa' in self.fusion_mode:
            out_2d = self.csa_blocks[idx](x2, x1) # order is 3d, 2d
            return out_2d
        elif self.fusion_mode == 'backup_ccnet_nonlocal':
            # this is a kind of position-agnostic attention between 2D and 3D features.
            n, c, d, h, w = x2.shape
            nl_feats = []
            for i in range(d):
                feat = self.nl_layers[idx](torch.cat((x1, x2[:, :, i, :, :]), dim=1)).unsqueeze(2)
                nl_feats.append(feat)
            nl_feats = torch.cat(nl_feats, dim=2)
            attention = self.fusion_layers[idx](nl_feats)
            n, c, d, h, w = attention.shape
            out_2d = x1 + attention[:, :, d//2, :, :]
            out_3d = x2 + nl_feats
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

