import logging
from abc import ABCMeta, abstractmethod

import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch.nn as nn
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
from mmdet.core import auto_fp16, get_classes, tensor2imgs
import json
import os


class BaseDetector(nn.Module):
    """Base class for detectors"""

    __metaclass__ = ABCMeta

    def __init__(self):
        super(BaseDetector, self).__init__()
        self.fp16_enabled = False

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_shared_head(self):
        return hasattr(self, 'shared_head') and self.shared_head is not None

    @property
    def with_bbox(self):
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_mask(self):
        return hasattr(self, 'mask_head') and self.mask_head is not None

    @abstractmethod
    def extract_feat(self, imgs):
        pass

    def extract_feats(self, imgs):
        assert isinstance(imgs, list)
        for img in imgs:
            yield self.extract_feat(img)

    @abstractmethod
    def forward_train(self, imgs, img_metas, **kwargs):
        """
        Args:
            img (list[Tensor]): list of tensors of shape (1, C, H, W).
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has:
                'img_shape', 'scale_factor', 'flip', and my also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

             **kwargs: specific to concrete implementation
        """
        pass

    @abstractmethod
    def simple_test(self, img, img_meta, **kwargs):
        pass

    @abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):
        pass

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            logger = logging.getLogger()
            logger.info('load model from: {}'.format(pretrained))

    def forward_test(self, imgs, img_metas, **kwargs):
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(imgs), len(img_metas)))
        # TODO: remove the restriction of imgs_per_gpu == 1 when prepared
        imgs_per_gpu = imgs[0].size(0)
        assert imgs_per_gpu == 1

        if num_augs == 1:
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            return self.aug_test(imgs, img_metas, **kwargs)

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_meta, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img, img_meta, **kwargs)
        else:
            return self.forward_test(img, img_meta, **kwargs)

    def show_result(self, data, result, save = None, dataset=None, score_thr=0.3, ann_file = False , json_file = False, coco=False,  vis_single_slice = False):
        if isinstance(result, tuple):
            bbox_result, segm_result = result
        else:
            bbox_result, segm_result = result, None

        img_tensor = data['img'][0]
        img_metas = data['img_meta'][0].data[0]
        imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
        assert len(imgs) == len(img_metas)

        if dataset is None:
            class_names = self.CLASSES
        elif isinstance(dataset, str):
            class_names = get_classes(dataset)
        elif isinstance(dataset, (list, tuple)):
            class_names = dataset
        else:
            raise TypeError(
                'dataset must be a valid dataset name or a sequence'
                ' of class names, not {}'.format(type(dataset)))

        for img, img_meta in zip(imgs, img_metas):
            h, w, _ = img_meta['img_shape']
            img_show = img[:h, :w, :]
            if vis_single_slice:
                img_show = img_show[:,:,1]
                img_show = img_show[:,:,np.newaxis]
                img_show = np.tile(img_show,(1,1,3))
            bboxes = np.vstack(bbox_result)
            # draw segmentation masks
            if segm_result is not None:
                segms = mmcv.concat_list(segm_result)
                inds = np.where(bboxes[:, -1] > score_thr)[0]
                for i in inds:
                    color_mask = np.random.randint(
                        0, 256, (1, 3), dtype=np.uint8)
                    mask = maskUtils.decode(segms[i]).astype(np.bool)
                    img_show[mask] = img_show[mask] * 0.5 + color_mask * 0.5
            # draw bounding boxes
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
            ]
            labels = np.concatenate(labels)
            
            if save: 
                if not os.path.exists(save):
                    os.mkdir(save)
                dpi = 100
                img_name = (img_metas[0]['filename'].split('/')[-4] + "_" + img_metas[0]['filename'].split('/')[-1])
                # patient name + image name
                img_show = img_show[:,:,::-1]
                scale_factor = img_meta['scale_factor']
                annFile = ann_file
                imgs = json_file['images']
                for i in range(len(imgs)):
                    if imgs[i]['file_name'] in img_metas[0]['filename']:
                        img_id = imgs[i]['id']
                annIds = coco.getAnnIds(img_id,iscrowd=None)
                anns = coco.loadAnns(annIds)
                
                fig = plt.figure(frameon=False)
                fig.set_size_inches(h / dpi, w / dpi)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.axis('off')
                fig.add_axes(ax)
                im = np.hstack([img_show,img_show])
                ax.imshow(im)
                
                for n in range(len(anns)):
                    x, y, w, h = anns[n]['bbox']
                    x, y, w, h = int(x * scale_factor), int(y * scale_factor), int(w * scale_factor), int(h * scale_factor)
                    label = anns[n]['category_id']

                    
                    for i in range(len(json_file['categories'])):
                        if json_file['categories'][i]['id'] == label:
                            label_text = json_file['categories'][i]['name']
                    ax.add_patch(
                              plt.Rectangle((x, y), w, h,
                              fill=False, edgecolor='g',
                              linewidth=1.0, alpha=1.0))
                    ax.text(x, y + h + 20, label_text,fontsize=10, 
                            family='sans-serif',bbox=dict(facecolor='g', alpha=0.4, pad=0, edgecolor='none'),color='white')
                    
                if score_thr > 0:
                    assert bboxes.shape[1] == 5
                    scores = bboxes[:, -1]
                    inds = scores > score_thr
                    bboxes = bboxes[inds, :]
                    labels = labels[inds]
                
                for bbox, label in zip(bboxes, labels):
                    score = bbox[4]
                    score = '%.2f' % score
                    bbox_int = (bbox* scale_factor).astype(np.int32)
                    x = bbox_int[0] 
                    y = bbox_int[1] 
                    w = (bbox_int[2] - bbox_int[0]) 
                    h = (bbox_int[3] - bbox_int[1]) 
                    
                    label_text = class_names[label] if class_names is not None else 'cls {}'.format(label)
                    ax.add_patch(
                              plt.Rectangle((x, y), w, h,
                              fill=False, edgecolor='r',
                              linewidth=1.0, alpha=1.0))
                    ax.text(x, y - 2, label_text + str(score),fontsize=10, 
                            family='sans-serif',bbox=dict(facecolor='r', alpha=0.4, pad=0, edgecolor='none'),color='white')
                    
                fig.savefig(save+ '/' + img_name)
                plt.close('all')
             
                
            else:
                mmcv.imshow_det_bboxes(
                    img_show,
                    bboxes,
                    labels,
                    class_names=class_names,
                    score_thr=score_thr,
                )
