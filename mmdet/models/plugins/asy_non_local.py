import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, normal_init

from ..utils import ConvModule


class AsyNonLocal2D(nn.Module):
    """Non-local module.

    See https://arxiv.org/abs/1711.07971 for details of non-local 2D
    See https://arxiv.org/pdf/1908.07678.pdf for details of asymetric non-local 
    

    Args:
        in_channels (int): Channels of the input feature map.
        reduction (int): Channel reduction ratio.
        use_scale (bool): Whether to scale pairwise_weight by 1/inter_channels.
        conv_cfg (dict): The config dict for convolution layers.
            (only applicable to conv_out)
        norm_cfg (dict): The config dict for normalization layers.
            (only applicable to conv_out)
        mode (str): Options are `embedded_gaussian` and `dot_product`.

        self.g is used for the reference input
        self.theta is used for the querry input
        self.phi is also used for the reference input
        cross_attention is conducted by: dot_product(self.theta, self.phi) * self.g
    """

    def __init__(self,
                 in_channels,
                 refer_in_channels,
                 reduction=2,
                 use_scale=True,
                 conv_cfg=None,
                 norm_cfg=None,
                 mode='embedded_gaussian'):
        super(AsyNonLocal2D, self).__init__()
        self.in_channels = in_channels
        self.refer_in_channels = refer_in_channels
        self.reduction = reduction
        self.use_scale = use_scale
        self.inter_channels = in_channels // reduction
        self.mode = mode
        assert mode in ['embedded_gaussian', 'dot_product']

        # g, theta, phi are actually `nn.Conv2d`. Here we use ConvModule for
        # potential usage.
        self.g = ConvModule(
            self.refer_in_channels,
            self.inter_channels,
            kernel_size=1,
            activation=None)
        self.theta = ConvModule(
            self.in_channels,
            self.inter_channels,
            kernel_size=1,
            activation=None)
        self.phi = ConvModule(
            self.refer_in_channels,
            self.inter_channels,
            kernel_size=1,
            activation=None)
        self.conv_out = ConvModule(
            self.inter_channels,
            self.in_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            activation=None)
        self.pooling = nn.MaxPool2d(kernel_size=2, padding=0, stride=2)
        self.init_weights()

    def init_weights(self, std=0.01, zeros_init=True):
        for m in [self.g, self.theta, self.phi]:
            normal_init(m.conv, std=std)
        if zeros_init:
            constant_init(self.conv_out.conv, 0)
        else:
            normal_init(self.conv_out.conv, std=std)

    def embedded_gaussian(self, theta_x, phi_x):
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        if self.use_scale:
            # theta_x.shape[-1] is `self.inter_channels`
            pairwise_weight /= theta_x.shape[-1]**0.5
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight

    def dot_product(self, theta_x, phi_x):
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight /= pairwise_weight.shape[-1]
        return pairwise_weight

    def forward(self, query, reference, downsample=False):
        '''
        ##forward by: dot_product(self.theta(q), self.phi(ref)) * self.g(ref)
        # default shape is(ignore channel reduction in conv theta phi and g):
        # theta(HW,C) * phi(C,DHW) * g(DHW,C)
        # if use CVPR2020 NL-reduction:
        # (HW,C/2) * (C/2,DHW/4) * g(DHW/4, C)
        '''
        rn, _, rh, rw = query.shape
        qn, _, qh, qw = reference.shape
        output = query
        if rh > 100 or rw > 100:
            downsample = True
            query = self.pooling(query)
            reference = self.pooling(reference)
            rn, _, rh, rw = query.shape
            qn, _, qh, qw = reference.shape

        # g_x: [N, DxH'xW', C] for reference 
        # reference in N C DH' W'; g(reference) in N C' DH' W';
        g_x = self.g(reference).view(rn, self.inter_channels, -1) # gx in N C' DH'W'
        g_x = g_x.permute(0, 2, 1) #gx in N DH'W' C'

        # theta_x: [N, HxW, C] for query
        theta_x = self.theta(query).view(qn, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        # phi_x: [N, C, DH'xW'] for reference
        phi_x = self.phi(reference).view(rn, self.inter_channels, -1) # phi_x in (N C' DH'W')

        pairwise_func = getattr(self, self.mode)
        # pairwise_weight: [N, HxW, DH'xW']
        pairwise_weight = pairwise_func(theta_x, phi_x)

        # y: [N, HxW, C]
        y = torch.matmul(pairwise_weight, g_x)
        # y: [N, C, H, W]
        y = y.permute(0, 2, 1).reshape(rn, self.inter_channels, rh, rw)

        # added by lzh
        if downsample:
            y = F.interpolate(y, scale_factor=2, mode='nearest')
        output = output + self.conv_out(y)
        #print('before add', y.shape, output.shape)
        #output = query + self.conv_out(y)

        return output
if __name__ == '__main__':
    n ,c, d ,h, w = [1, 64, 3, 144, 144]
    x = torch.rand((n ,c ,h, w))
    y = torch.rand((n ,c, d ,h, w))
    y = y.view(1,-1, h, w)
    asy_nl = AsyNonLocal2D(64, 64)




