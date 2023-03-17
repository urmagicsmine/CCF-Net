import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.core import auto_fp16
from ..registry import NECKS
from ..utils import ConvModule
from .cbam import *


@NECKS.register_module
class MSB_FPN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None):
        super(MSB_FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        #self.aspps = nn.ModuleList()
        #self.cbams = nn.ModuleList()
        #for i in range(self.num_outs):
            #self.cbams.append(CBAM(256, 16))
            #self.aspps.append(ASPP(256, 256, [1,2,3]))
        self.cbam = CBAM(256, 16)
        self.aspp = ASPP(256, 256, [1,2,3])

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                activation=self.activation,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    activation=self.activation,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=2, mode='nearest')

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                else:
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        msb_outs = []
        for i in range(self.num_outs):
            #msb_outs.append(self.cbams[i](self.aspps[i](outs[i])))
            #msb_outs.append(self.cbams[i](outs[i]))
            #msb_outs.append(self.aspps[i](outs[i]))
            msb_outs.append(self.cbam(self.aspp(outs[i])))
        return tuple(msb_outs)
        #return tuple(outs)

class ASPP(nn.Module):
    """ Fuse infomation from different rated astrous_convs
        Will reduce to the original posthoc implementation if len(dilation_list) = 1
    """
    def __init__(self, inplanes, planes, dilation_list):
        super(ASPP, self).__init__()
        if not isinstance(dilation_list, list):
            self.dilation_list = [dilation_list]
        else:
            self.dilation_list = dilation_list
        # Add resnet_block-like constructure(channel 256->64->256) to reduce computation consume.
        if len(dilation_list) ==3:
            self.aspp1 = _atconv(inplanes, 256, 3, padding=dilation_list[0], dilation=dilation_list[0])
            self.aspp2 = _atconv(inplanes, 256, 3, padding=dilation_list[1], dilation=dilation_list[1])
            self.aspp3 = _atconv(inplanes, 256, 3, padding=dilation_list[2], dilation=dilation_list[2])
            #self.aspp4 = _atconv(inplanes, 64, 1, padding=0, dilation=1)
            self.aspp4 = nn.Conv2d(inplanes, 256, 1, bias=False)
            self.aspp_conv1 = nn.Conv2d(256* 4, planes, 1, bias=False)
            #self.aspp_conv1 = nn.Conv2d(64 * 3, planes, 1, bias=False)
            #self.dropout = nn.Dropout(0.5)
            self._init_weight()
        elif len(dilation_list) ==1:
            # If dilation_list has only one element [1], aspp is the same as the original posthoc.
            self.aspp1 = _atconv(inplanes, planes, 3, padding=dilation_list[0], dilation=dilation_list[0])
        else:
            raise NotImplementedError


    def forward(self, x):
        if len(self.dilation_list) == 3:
            x1 = self.aspp1(x)
            x2 = self.aspp2(x)
            x3 = self.aspp3(x)
            x4 = self.aspp4(x)
            x = torch.cat((x1, x2, x3,x4), dim=1)
            #x = torch.cat((x1, x2, x3), dim=1)
            x = self.aspp_conv1(x)
            return x
        elif len(self.dilation_list) == 1:
            x = self.aspp1(x)
        if len(self.dilation_list) == 3:
            return self.dropout(x)
        elif len(self.dilation_list) == 1:
            return x

    def _init_weight(self):
        xavier_init(self.aspp_conv1, distribution='uniform')
        xavier_init(self.aspp4, distribution='uniform')
        #conv = self.aspp_conv1
        #if cfg.FPN.ZERO_INIT_LATERAL:
            #init.constant_(conv.weight, 0)
        #else:
            #mynn.init.XavierFill(conv.weight)
        #if conv.bias is not None:
            #init.constant_(conv.bias, 0)


class  _atconv(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_atconv, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes,
                kernel_size=kernel_size, stride=1, padding=padding,
                dilation=dilation, bias=False)
        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        return x

    def _init_weight(self):
        xavier_init(self.atrous_conv, distribution='uniform')
        #conv = self.atrous_conv
        #if cfg.FPN.ZERO_INIT_LATERAL:
            #init.constant_(conv.weight, 0)
        #else:
            #mynn.init.XavierFill(conv.weight)
        #if conv.bias is not None:
            #init.constant_(conv.bias, 0)

