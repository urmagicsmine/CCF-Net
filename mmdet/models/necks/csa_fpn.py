import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.core import auto_fp16
from ..registry import NECKS
from ..utils import ConvModule
from ..backbones.cross_layer_fusion import CSABlock


@NECKS.register_module
class CSA_FPN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 cross_slice_attention_type='nonlocal',
                 attention_before_fpnconv = False,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None):
        super(CSA_FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.cross_slice_attention_type = cross_slice_attention_type
        self.attention_before_fpnconv = attention_before_fpnconv
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
        self.CSABlocks = nn.ModuleList()
        fpn_conv_cfg = None if self.attention_before_fpnconv else conv_cfg
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                activation=self.activation,
                inplace=False)
            #csa_channels = out_channels \
                #if 'ori' in self.cross_slice_attention_type else 256
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                #conv_cfg=conv_cfg,
                conv_cfg=fpn_conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)
            csa_block = CSABlock(out_channels, out_channels//2,
                    self.cross_slice_attention_type)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            self.CSABlocks.append(csa_block)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1: #fasle for faster rcnn
            for i in range(extra_levels):
                if i == 0 and self.gxtra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    #conv_cfg=conv_cfg,
                    conv_cfg=fpn_conv_cfg,
                    norm_cfg=norm_cfg,
                    activation=self.activation,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)
                # add extra CSA Blocks .
                csa_block = CSABlock(256, 128, self.cross_slice_attention_type)
                self.CSABlocks.append(csa_block)
        if self.num_outs > len(self.lateral_convs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN) (5 > 4)
            # so add an extra CSA Block too.
            if not self.add_extra_convs:
                csa_block = CSABlock(256, 128, cross_slice_attention_type)
                self.CSABlocks.append(csa_block)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        #for item in inputs:
            #print('input',item.shape)
        #for item in laterals:
            #print('lateral',item.shape)
        # build top-down path
        used_backbone_levels = len(laterals)
        # TODO: check nearest interpolation
        if 'ori' in self.cross_slice_attention_type:
            for i in range(used_backbone_levels - 1, 2, -1): # 2345
                laterals[i - 1] += F.interpolate(
                    self.CSABlocks[i](laterals[i]), scale_factor=(1,2,2), mode='nearest')
            for i in range(2, 0, -1):
                laterals[i - 1] += F.interpolate(
                    laterals[i], scale_factor=(1,2,2), mode='nearest')
        else:
            for i in range(used_backbone_levels - 1, 0, -1):
                laterals[i - 1] += F.interpolate(
                    laterals[i], scale_factor=(1,2,2), mode='nearest')
                    #laterals[i], scale_factor=2, mode='trilinear')

        # Type1: CSABlock before 3*3 fpn conv.
        if self.attention_before_fpnconv == True:
            middle_features = self.process_csa(laterals, skip_blocks=1)
        else:
            middle_features = laterals

        # build outputs
        # part 1: from original levels
        outs = [
            #self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
            self.fpn_convs[i](middle_features[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool3d(outs[-1], 1, stride=(1,2,2)))
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
        #print(len(outs), used_backbone_levels) # 5 and 4
        # Type2: CSABlock after 3*3 fpn conv.
        if self.attention_before_fpnconv == False:
            final_outs = self.process_csa(outs, skip_blocks=2)
        elif self.attention_before_fpnconv == None:
            # if donot apply CSAblock, features in outs is 3D(rather than 2D)
            # so we need to take center slice here.
            N, C, D, H, W = outs[0].shape
            center = D // 2
            final_outs = [out[:, :, center, :, :] for out in outs]
        else:
            final_outs = outs
        return tuple(final_outs)

    def process_csa(self, features, skip_blocks):
        ''''
        Applying CSABlocks while skip prior num=$skip_blocks$ stages.
        In these skipped stage, features of center slice will be taken.
        CSABlocks are used in later stages.
        '''
        if self.cross_slice_attention_type in ['nonlocal', 'nonlocalori']:
            N, C, D, H, W = features[0].shape
            outs = [ features[i][:,:, D//2, :, :] for i in range(0, skip_blocks) ]
            outs.extend([
                self.CSABlocks[i](features[i]) for i in range(skip_blocks, len(features))
            ])

        #elif self.cross_slice_attention_type == 'maxpool' or \
            #self.cross_slice_attention_type == 'traverse':
        else:
            outs=[ self.CSABlocks[i](features[i]) for i in range(len(features)) ]
        #for i in range(len(features)):
            #print(features[i].shape, outs[i].shape)
        return outs
