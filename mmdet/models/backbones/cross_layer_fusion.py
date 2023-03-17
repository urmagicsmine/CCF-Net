''' This file includes the implementation of cross layer fusion, which is used to fuse multi-slice feature in faster-rcnn-mp3d backbone
    Some funcions are copied from 'fusion.py'
'''

import numpy as np
from scipy import ndimage
import os
import cv2
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import pdb

class NL_Module(nn.Module):
    """ Spatial attention module (similar to Non-Local)
    Not Used here.
    """
    def __init__(self, in_dim):
        super(NL_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x, y):
        """
            inputs :
                x : input feature maps( B X C X H X W )
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        # (HW,C)
        proj_query = x.view(m_batchsize, C, -1).permute(0, 2, 1).contiguous()
        # (C,HW)
        proj_key = y.view(m_batchsize, C, -1)
        # (HW, HW)
        f = torch.bmm(proj_query, proj_key)
        #f_new = torch.max(f, -1, keepdim=True)[0].expand_as(f)-f
        f_new = torch.max(f, -1, keepdim=True)[0].expand_as(f)-f
        attention = self.softmax(f_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma*out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module """
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W )
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        # (C,HW)
        proj_query = x.view(m_batchsize, C, -1)
        # (HW,C)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        # (C,C)
        energy = torch.bmm(proj_query, proj_key)
        pdb.set_trace()
        tmp = torch.max(energy, -1, keepdim=True)
        print(tmp.shape)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        print(energy_new.shape)
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class CFAMBlock(nn.Module):
    def __init__(self, in_channels, inter_channels):
        super(CFAMBlock, self).__init__()
        self.conv_bn_relu1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())
        self.conv_bn_relu2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())

        self.sc = SAM_Module(inter_channels)

        self.conv_bn_relu3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU())

        self.conv_out = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x, y):
        n, c, d, h, w = x.shape
        x = x.view(n,)
        x = self.conv_bn_relu1(x)
        x = self.conv_bn_relu2(x)
        x = self.sc(x)
        x = self.conv_bn_relu3(x)
        output = self.conv_out(x)

        return output

# Cross Slice Attention
class CSABlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode='maxpool'):
        super(CSABlock, self).__init__()

        self.inter_channels = in_channels//2 if inter_channels==None else inter_channels
        ##assert mode in ['maxpool', 'traverse', 'nonlocal',
                ##'reversemaxpool', 'qrqmaxpool', 'center']
        # temp
        if mode == 'maxpool_bnrelu':
            temp_mode = mode
            mode = 'maxpool'
        elif mode == 'maxpool_afnb':
            temp_mode = mode
            mode = 'maxpool'
        else:
            temp_mode = None
        self.mode = mode
        if mode == 'nonlocal' or mode == 'nonlocalori' or mode=='channel_attention':
            conv_function = nn.Conv3d
            bn_function = nn.BatchNorm3d
        else:
            conv_function = nn.Conv2d
            bn_function = nn.BatchNorm2d
        if temp_mode == 'maxpool_bnrelu':
            self.theta = nn.Sequential(conv_function(in_channels, inter_channels, kernel_size=1, padding=0, bias=False),
                                    bn_function(inter_channels), nn.ReLU()
                                    )
            self.phi = nn.Sequential(conv_function(in_channels, inter_channels, kernel_size=1, padding=0, bias=False),
                                    bn_function(inter_channels), nn.ReLU()
                                    )

            self.g = nn.Sequential(conv_function(in_channels, inter_channels, kernel_size=1, padding=0, bias=False),
                                   bn_function(inter_channels), nn.ReLU()
                                   )
            self.w = nn.Sequential(conv_function(inter_channels, in_channels, kernel_size=1, padding=0, bias=False),
                                   bn_function(in_channels),
                                   nn.ReLU()
                                   )
        elif temp_mode == 'maxpool_afnb':
            self.theta = nn.Sequential(conv_function(in_channels, inter_channels, kernel_size=1, padding=0, bias=False),
                                    bn_function(inter_channels), nn.ReLU()
                                    )
            self.phi = nn.Sequential(conv_function(in_channels, inter_channels, kernel_size=1, padding=0, bias=False),
                                    bn_function(inter_channels), nn.ReLU()
                                    )

            self.g = nn.Sequential(conv_function(in_channels, inter_channels, kernel_size=1, padding=0, bias=False),
                                   #bn_function(inter_channels), nn.ReLU()
                                   )
            self.w = nn.Sequential(conv_function(inter_channels, in_channels, kernel_size=1, padding=0, bias=False),
                                   #bn_function(in_channels), nn.ReLU()
                                   )
        else:
            self.theta = nn.Sequential(conv_function(in_channels, inter_channels, kernel_size=1, padding=0, bias=False),
                                    bn_function(inter_channels), nn.ReLU()
                                    )
            self.phi = nn.Sequential(conv_function(in_channels, inter_channels, kernel_size=1, padding=0, bias=False),
                                    bn_function(inter_channels), nn.ReLU()
                                    )

            self.g = nn.Sequential(conv_function(in_channels, inter_channels, kernel_size=1, padding=0, bias=False),
                                   #bn_function(inter_channels), nn.ReLU()
                                   )
            self.w = nn.Sequential(conv_function(inter_channels, in_channels, kernel_size=1, padding=0, bias=False),
                                   #bn_function(in_channels),
                                   #nn.ReLU()
                                   )
        if 'csa' in self.mode:
            conv_function = nn.Conv2d
            bn_function = nn.BatchNorm2d
            self.theta = nn.Sequential(conv_function(in_channels, inter_channels, kernel_size=1, padding=0, bias=False),
                                    bn_function(inter_channels), nn.ReLU()
                                    )
            self.w = nn.Sequential(conv_function(inter_channels, in_channels, kernel_size=1, padding=0, bias=False),
                                   )
            conv_function = nn.Conv3d
            bn_function = nn.BatchNorm3d
            self.phi = nn.Sequential(conv_function(in_channels, inter_channels, kernel_size=1, padding=0, bias=False),
                                    bn_function(inter_channels), nn.ReLU()
                                    )

            self.g = nn.Sequential(conv_function(in_channels, inter_channels, kernel_size=1, padding=0, bias=False),
                                    #bn_function(inter_channels), nn.ReLU()
                                   )
        if self.mode == 'qrqmaxpool':
            self.alpha = nn.Sequential(nn.Conv2d(inter_channels * 2, inter_channels, kernel_size=1, padding=0, bias=False),
                                   bn_function(in_channels),
                                   )


    def forward(self, feature, center=None):
        N, C, D, H, W = feature.shape
        center_x = feature[:, :, D//2+1, :, :]
        if self.mode == 'maxpool':
            #deprocess_image(center_x, 'beforemaxppol')
            x = F.max_pool3d(feature, kernel_size=(D, 1, 1), stride=(1, 1, 1)).view(N, C, H, W)
            # (HW, C) query, center
            theta_x = self.theta(center_x)
            theta_x = theta_x.view(N, self.inter_channels, -1).permute(0, 2, 1)
            # (C, HW) reference, arbitrary slice
            phi_x = self.phi(x).view(N, self.inter_channels, -1)
            # (HW, C) reference
            g_x = self.g(x).view(N, self.inter_channels, -1).permute(0, 2, 1)
            # (HW, HW) weight matrix
            f = torch.matmul(theta_x, phi_x)
            weight = F.softmax(f, dim=-1)
            # (HW, C)
            weighted_x = torch.matmul(weight, g_x)
            weighted_x = weighted_x.permute(0,2,1).view(N, self.inter_channels, H, W)
            out = self.w(weighted_x)
            #deprocess_image(out, 'diff')
            out = out + center_x
            #deprocess_image(out, 'aftermaxppoltest')

        elif self.mode == 'ccnet_maxpool':
            pass
        elif self.mode == 'reversemaxpool':
            x = F.max_pool3d(feature, kernel_size=(D, 1, 1), stride=(1, 1, 1)).view(N, C, H, W)
            # (C, HW) query, center
            theta_x = self.theta(center_x)
            theta_x = theta_x.view(N, self.inter_channels, -1)
            # (HW, C) reference, arbitrary slice
            phi_x = self.phi(x).view(N, self.inter_channels, -1).permute(0, 2, 1)
            # (HW, C) reference
            g_x = self.g(center_x).view(N, self.inter_channels, -1).permute(0, 2, 1)
            # (HW, HW) weight matrix
            f = torch.matmul(phi_x, theta_x)
            weight = F.softmax(f, dim=-1)
            # (HW, C)
            weighted_x = torch.matmul(weight, g_x)
            weighted_x = weighted_x.permute(0,2,1).view(N, self.inter_channels, H, W)
            out = self.w(weighted_x)
            out = out + center_x

        elif self.mode == 'centeronly':
            return center_x

        elif self.mode == 'maxpoolonly':
            x = F.max_pool3d(feature, kernel_size=(D, 1, 1), stride=(1, 1, 1)).view(N, C, H, W)
            return x

        elif self.mode == 'cm':
            x = F.max_pool3d(feature, kernel_size=(D, 1, 1), stride=(1, 1, 1)).view(N, C, H, W)
            return x + center_x

        elif self.mode == 'qrqmaxpool':
            x = F.max_pool3d(feature, kernel_size=(D, 1, 1), stride=(1, 1, 1)).view(N, C, H, W)
            # (C, HW) query, center
            theta_x = self.theta(center_x)
            theta_x = theta_x.view(N, self.inter_channels, -1)
            # (HW, C) reference, arbitrary slice
            phi_x = self.phi(x).view(N, self.inter_channels, -1).permute(0, 2, 1)
            # (HW, C) reference
            g_x = self.g(center_x).view(N, self.inter_channels, -1).permute(0, 2, 1)
            # (HW, HW) weight matrix
            f = torch.matmul(phi_x, theta_x)
            weight = F.softmax(f, dim=-1)
            concat_weight = torch.cat((weight, weight.permute(0,2,1)), 2).unsqueeze(1)
            print(concat_weight.shape)
            new_weight = self.alpha(concat_weight).squeeze(1)
            # (HW, C)
            weighted_x = torch.matmul(new_weight, g_x)
            weighted_x = weighted_x.permute(0,2,1).view(N, self.inter_channels, H, W)
            out = self.w(weighted_x)
            out = out + center_x

        elif self.mode == 'traverse':
            out = center_x + 0 # deepcopy
            for depth in range(D):
                if depth == D//2: #skip center slice itself.
                    continue
                x = feature[:, :, depth, :, :]
                # (HW, C) query, center
                theta_x = self.theta(center_x)
                theta_x = theta_x.view(N, self.inter_channels, -1).permute(0, 2, 1)
                # (C, HW) reference, arbitrary slice
                phi_x = self.phi(x).view(N, self.inter_channels, -1)
                # (HW, C) reference
                g_x = self.g(x).view(N, self.inter_channels, -1).permute(0, 2, 1)
                # (HW, HW) weight matrix
                f = torch.matmul(theta_x, phi_x)
                weight = F.softmax(f, dim=-1)
                # (HW, C)
                weighted_x = torch.matmul(weight, g_x)
                weighted_x = weighted_x.permute(0,2,1).view(N, self.inter_channels, H, W)
                out += self.w(weighted_x)
        elif self.mode == 'nonlocal' or self.mode == 'nonlocalori':
            #deprocess_image(center_x, 'beforenonlocal')
            # TODO: need conv3d
            # (DHW, C) query, center
            x = feature
            theta_x = self.theta(x)
            theta_x = theta_x.view(N, self.inter_channels, -1).permute(0, 2, 1)
            # (C, DHW) reference, arbitrary slice
            phi_x = self.phi(x).view(N, self.inter_channels, -1)
            # (DHW, C) reference
            g_x = self.g(x).view(N, self.inter_channels, -1).permute(0, 2, 1)
            # (DHW, DHW) weight matrix
            f = torch.matmul(theta_x, phi_x)
            weight = F.softmax(f, dim=-1)
            # (DHW, C)
            weighted_x = torch.matmul(weight, g_x)
            weighted_x = weighted_x.permute(0,2,1).view(N, self.inter_channels, D, H, W)
            #print(self.w)
            #print(weighted_x.shape)
            # TODO: out+= or  out= ?
            if self.mode == 'nonlocal':
                #out = self.w(weighted_x)[:, :, D//2, :, :]
                out = (x + self.w(weighted_x))[:, :, D//2, :, :]
                #deprocess_image(weighted_x[:,:,D//2,:,:], 'diffnonlocal')
            elif self.mode == 'nonlocalori':
                out = self.w(weighted_x)
            #deprocess_image(weighted_x[:,:,D//2,:,:], 'afternonlocal')
        elif self.mode == 'channel_attention':
            x = feature
            theta_x = self.theta(x)
            # (C, DHW) reference
            theta_x = theta_x.view(N, self.inter_channels, -1).permute(0, 2, 1)
            # (DHW, C) reference
            phi_x = self.phi(x).view(N, self.inter_channels, -1)
            # (C, DHW) reference
            g_x = self.g(x).view(N, self.inter_channels, -1) #.permute(0, 2, 1)
            # (C, C) weight matrix
            f = torch.matmul(phi_x, theta_x)
            f_new = torch.max(f, -1, keepdim=True)[0].expand_as(f)-f
            weight = F.softmax(f_new, dim=-1)
            # (DHW, C)
            weighted_x = torch.matmul(weight, g_x)
            #tmp = torch.max(weighted_x, -1, keepdim=True)
            #energy_new = torch.max(weighted_x, -1, keepdim=True)[0].expand_as(weighted_x)-weighted_x
            weighted_x = weighted_x.view(N, self.inter_channels, D, H, W)
            out = (x + self.w(weighted_x))[:, :, D//2, :, :]

            #m_batchsize, C, height, width = x.size()
            ## (C,HW)
            #proj_query = x.view(m_batchsize, C, -1)
            ## (HW,C)
            #proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
            ## (C,C)
            #energy = torch.bmm(proj_query, proj_key)
            #pdb.set_trace()
            #tmp = torch.max(energy, -1, keepdim=True)
            #energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
            #print(energy_new.shape)
            #attention = self.softmax(energy_new)
            #proj_value = x.view(m_batchsize, C, -1)
            #out = torch.bmm(attention, proj_value)
            #out = out.view(m_batchsize, C, height, width)
            #out = self.gamma*out + x
            #return out


        elif self.mode == 'channel-nonlocal':
            # TODO: need conv3d
            # (DHW, C) query, center
            x = feature
            theta_x = self.theta(x)
            theta_x = theta_x.view(N, self.inter_channels, -1).permute(0, 2, 1)
            # (C, DHW) reference, arbitrary slice
            phi_x = self.phi(x).view(N, self.inter_channels, -1)
            # (DHW, C) reference
            g_x = self.g(x).view(N, self.inter_channels, -1).permute(0, 2, 1)
            # (DHW, DHW) weight matrix
            f = torch.matmul(theta_x, phi_x)
            weight = F.softmax(f, dim=-1)
            # (DHW, C)
            weighted_x = torch.matmul(weight, g_x)
            weighted_x = weighted_x.permute(0,2,1).view(N, self.inter_channels, D, H, W)
            #print(self.w)
            #print(weighted_x.shape)
            out = self.w(weighted_x)[:, :, D//2, :, :]
        elif self.mode == 'csa_nonlocal':
            # (DHW, C) query, center
            x = feature
            theta_x = self.theta(center)
            theta_x = theta_x.view(N, self.inter_channels, -1).permute(0, 2, 1)
            #print(self.inter_channels, center.shape, feature.shape, x.shape, self.phi)
            # (C, DHW) reference, arbitrary slice
            phi_x = self.phi(x).view(N, self.inter_channels, -1)
            # (DHW, C) reference
            g_x = self.g(x).view(N, self.inter_channels, -1).permute(0, 2, 1)
            # (DHW, DHW) weight matrix
            f = torch.matmul(theta_x, phi_x)
            weight = F.softmax(f, dim=-1)
            # (DHW, C)
            weighted_x = torch.matmul(weight, g_x)
            weighted_x = weighted_x.permute(0,2,1).view(N, self.inter_channels, H, W)
            #print(self.w)
            #print(weighted_x.shape)
            out = center + self.w(weighted_x)
        elif self.mode == 'center':
            return center_x
        else:
            raise NotImplementedError('cross_slice_attention.py: this func is not impled')

        return out

def deprocess_image(feature, name='default', k=0):
    target_size=(512,512)
    print(feature.shape)
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
    save_dir = "./feature_image/avg_wo_bnrelu/"
    k = len(os.listdir(save_dir))
    save_path = save_dir  + str(k) + "_" + name + ".jpg"
    state = cv2.imwrite(save_path,image)
    print(state, k, save_path, img.shape)

if __name__ == "__main__":
    x = torch.randn(18, 64, 9, 7, 7) #.cuda()
    in_channels = x.size()[1]
    inter_channels = in_channels // 2
    #model = CSABlock(in_channels, inter_channels, 'maxpool')
    #model = CSABlock(in_channels, inter_channels, 'nonlocal')
    #model = CSABlock(in_channels, inter_channels, 'traverse').cuda()
    #model = CSABlock(in_channels, inter_channels, 'reversemaxpool').cuda()
    model = CSABlock(in_channels, inter_channels, 'channel_attention') #.cuda()
    output = model(x)
    #print(model)
    print(output.size())
