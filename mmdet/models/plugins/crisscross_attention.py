import torch
from torch.nn import functional as F
import torch.nn as nn
import cv2
import os
import numpy as np
from mmcv.cnn import constant_init, normal_init
from ..utils import ConvModule

from .cc_attention import CrissCrossAttention
from ..losses import EntropyLoss
#try:
    #from mmcv.ops import CrissCrossAttention
#except ModuleNotFoundError:
    #CrissCrossAttention = None


class CCAttention(nn.Module):
    """CCNet: Criss-Cross Attention for Semantic Segmentation.

    This head is the implementation of `CCNet
    <https://arxiv.org/abs/1811.11721>`_.

    Args:
        recurrence (int): Number of recurrence of Criss Cross Attention
            module. Default: 2.
    """

    def __init__(self, in_channels, inter_channels, recurrence=2, **kwargs):
        if CrissCrossAttention is None:
            raise RuntimeError('Please install mmcv-full for '
                               'CrissCrossAttention ops')
        super(CCAttention, self).__init__()
        self.recurrence = recurrence
        self.in_channels = in_channels
        self.inter_channels = inter_channels if inter_channels else in_channels // 2
        self.cca = CrissCrossAttention(self.inter_channels)
        self.conv1 = ConvModule(self.in_channels,
            self.inter_channels,
            kernel_size=1,
            padding=0,
            #norm_cfg = dict(type='GN', num_groups=32, requires_grad=True),
            activation=None)
        #self.conv2 = ConvModule(
            #self.inter_channels,
            #self.inter_channels,
            #kernel_size=3,
            #padding=1,
            #activation=None)
        self.conv_out = ConvModule(
            self.inter_channels,
            self.inter_channels,
            kernel_size=3,
            padding=1,
            #conv_cfg=conv_cfg,
            norm_cfg = dict(type='GN', num_groups=32, requires_grad=True),
            #norm_cfg=norm_cfg,
            activation=None)
        self.loss_fuc = EntropyLoss(loss_weight=0.5)

    def init_weights(self, std=0.01, zeros_init=True):
        for m in [self.conv1, self.conv2, self.conv_out]:
            normal_init(m.conv, std=std)
        if zeros_init:
            constant_init(self.conv_out.conv, 0)
        else:
            normal_init(self.conv_out.conv, std=std)

    def forward(self, x, ref=None, loss=[]):
        """Forward function."""
        save = False
        output = self.conv1(x)
        #if save:
            #vis_list = []
            #vis_list.append(deprocess_image(output))
        for i in range(self.recurrence):
            #output, camap = self.cca(output, ref)
            output = self.cca(output, ref)
            if save==True:
                vis_list.append(deprocess_image(camap))
        # attention map constraint, entropy loss
        #loss.append(self.loss_fuc(camap+0))
        #if save:
            #vis_list.append(deprocess_image(output))
            #concat_and_save(vis_list)
        #if self.concat_input:
            #output = self.conv_out(torch.cat([x, output], dim=1))
        #output = self.conv2(output)
        output = self.conv_out(output)
        return output

def deprocess_image(feature):
    target_size=(512,512)
    #print(feature.shape)
    n, c, h, w = feature.shape
    # skip small image
    #if h < 30:
        #return
    #img = F.max_pool3d(feature.data, kernel_size=(c,1,1), stride=(1,1,1)).cpu().view(-1, w).numpy()
    img = F.avg_pool3d(feature.data, kernel_size=(c,1,1), stride=(1,1,1)).cpu().view(-1, w).numpy()
    #img = cv2.resize(img,(512,512),interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32)
    img = cv2.resize(img,None,fx=2,fy=2)
    #img = ndimage.gaussian_filter(img, sigma=3)
    #img = img - np.mean(img)
    #img = img / (np.std(img) + 1e-5)
    #img = img * 0.1
    #img = img + 0.5
    img = (img-img.min()) / (img.max()-img.min())
    img = np.clip(img, 0, 1)
    img = np.uint8(img*255)
    image = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return image

def concat_and_save(image_list):
    #assert len(image_list) == 3
    h, w, c = image_list[0].shape
    blank = np.zeros((h, 10, c)).astype(np.uint8)
    concat_list = []
    for image in image_list:
        concat_list.append(image)
        concat_list.append(blank)
    concat_img = np.hstack(concat_list)
    save_image(concat_img)


def save_image(image, save_dir='./feature_image/'):
    k = len(os.listdir(save_dir))
    save_path = save_dir  + str(k) + ".jpg"
    state = cv2.imwrite(save_path,image)
    print(state, k, save_path, image.shape)

