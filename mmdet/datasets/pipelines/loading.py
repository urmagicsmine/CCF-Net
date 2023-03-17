import os.path as osp
import warnings
import cv2

import mmcv
import numpy as np
import pycocotools.mask as maskUtils

from ..registry import PIPELINES
from .image_io import load_multislice_gray_png, load_data
from .image_io_3DCE import load_multislice_gray_png_3DCE

@PIPELINES.register_module
class LoadImageFromFile(object):
    # Added to support multi-slice input by deepwise.
    def __init__(self, to_float32=False, lung_input=False, num_slice=3, zflip=False, med_view=False,lung_mip=False, extra_transform=False, zaug=False):
        self.to_float32 = to_float32
        self.lung_input = lung_input
        self.num_slice = num_slice
        self.med_view = med_view
        self.lung_mip = lung_mip
        self.extra_transform = extra_transform
        self.zaug = zaug
        if self.lung_mip:
            self.zflip = False
        else:
            self.zflip = zflip


    def __call__(self, results):
        filename = osp.join(results['img_prefix'],
                            results['img_info']['filename'])
        if not self.lung_input:
            img = mmcv.imread(filename)
        else:
            if self.med_view:
                #eg.lung_det_stage2_batch123->lung_det_stage2_batch123_med
                repl = results['img_prefix'].split('/')[-3]
                img_prefix_med = results['img_prefix'].replace(repl,repl+'_med')
                filename_med = osp.join(img_prefix_med,
                            results['img_info']['filename'])
                img1 = load_multislice_gray_png(filename, self.num_slice, self.zflip)
                img2 = load_multislice_gray_png(filename_med, self.num_slice, self.zflip)
                img = np.concatenate((img1,img2),axis=2)

            else:
                img = load_multislice_gray_png(filename, self.num_slice, self.zflip, self.lung_mip, self.extra_transform, self.zaug)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(to_float32={})'.format(
            self.to_float32)

@PIPELINES.register_module
class LoadImageFromFile_3DCE(object):
    # Added to support multi-slice input by deepwise.
    def __init__(self, to_float32=False, lesion_input=True, num_slice=9, zflip=False, lung_mip=False, multi_view=False, window=(-1024,1050)):
        self.to_float32 = to_float32
        self.lesion_input = lesion_input
        self.zflip = zflip
        self.num_slice = num_slice
        self.lung_mip = lung_mip
        self.multi_view = multi_view
        self.window = window

    def __call__(self, results):
        filename = osp.join(results['img_prefix'],
                            results['img_info']['filename'])
        slice_intv = results['img_info']['slice_intv']
        if not self.lesion_input:
            img = mmcv.imread(filename)
        else:
            img = load_multislice_gray_png_3DCE(filename, self.num_slice, slice_intv, self.zflip, self.lung_mip, self.multi_view, self.window)
        if self.to_float32:
            img = img.astype(np.float32)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        results['filename'] = filename
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(to_float32={})'.format(
            self.to_float32)

@PIPELINES.register_module
class LoadImageFromTensor(object):
    def __init__(self, to_float32=False, num_slice=3, window=None):
        ''' If 'window' is specified, process image npz as uint16 data type.
        '''
        self.to_float32 = to_float32
        self.num_slice = num_slice
        self.window = window

    def __call__(self, results):
        center_idx = results['img_info']['slice_index']
        image_tensor = results['image_tensor']
        im_cur = image_tensor[center_idx, :,:]
        ims = [im_cur]
        # find neighboring slices of im_cure
        rel_pos = 1
        for p in range((self.num_slice-1)//2):
            im_prev = load_data(image_tensor, center_idx, - rel_pos * (p + 1))
            im_next = load_data(image_tensor, center_idx, rel_pos * (p + 1))
            ims = [im_prev] + ims + [im_next]
        img = cv2.merge(ims)
        if self.to_float32:
            img = img.astype(np.float32)
        if not self.window is None:
            img = uint16_windowing(img, self.window)
        results['slice_index'] = center_idx
        results['filename'] = center_idx
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(to_float32={})'.format(
            self.to_float32)

def uint16_windowing(im, win):
    # Scale intensity from win[0]~win[1] to float numbers in 0~255
    im1 = im.astype(np.float32)
    im1 -= 32768
    im1 -= win[0]
    im1 /= win[1] - win[0]
    im1[im1 > 1] = 1
    im1[im1 < 0] = 0
    im1 *= 255
    return im1

@PIPELINES.register_module
class LoadAnnotations(object):

    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=False,
                 with_seg=False,
                 poly2mask=True,
                 skip_img_without_anno=True):
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.poly2mask = poly2mask
        self.skip_img_without_anno = skip_img_without_anno

    def _load_bboxes(self, results):
        ann_info = results['ann_info']
        results['gt_bboxes'] = ann_info['bboxes']
        if len(results['gt_bboxes']) == 0 and self.skip_img_without_anno:
            if results['img_prefix'] is not None:
                file_path = osp.join(results['img_prefix'],
                                     results['img_info']['filename'])
            else:
                file_path = results['img_info']['filename']
            warnings.warn(
                'Skip the image "{}" that has no valid gt bbox'.format(
                    file_path))
            return None
        results['gt_bboxes_ignore'] = ann_info.get('bboxes_ignore', None)
        results['bbox_fields'].extend(['gt_bboxes', 'gt_bboxes_ignore'])
        return results

    def _load_labels(self, results):
        results['gt_labels'] = results['ann_info']['labels']
        return results

    def _poly2mask(self, mask_ann, img_h, img_w):
        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def _load_masks(self, results):
        h, w = results['img_info']['height'], results['img_info']['width']
        gt_masks = results['ann_info']['masks']
        if self.poly2mask:
            gt_masks = [self._poly2mask(mask, h, w) for mask in gt_masks]
        results['gt_masks'] = gt_masks
        results['mask_fields'].append('gt_masks')
        return results

    def _load_semantic_seg(self, results):
        results['gt_semantic_seg'] = mmcv.imread(
            osp.join(results['seg_prefix'], results['ann_info']['seg_map']),
            flag='unchanged').squeeze()
        return results

    def __call__(self, results):
        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)
        if self.with_seg:
            results = self._load_semantic_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(with_bbox={}, with_label={}, with_mask={},'
                     ' with_seg={})').format(self.with_bbox, self.with_label,
                                             self.with_mask, self.with_seg)
        return repr_str


@PIPELINES.register_module
class LoadProposals(object):

    def __init__(self, num_max_proposals=None):
        self.num_max_proposals = num_max_proposals

    def __call__(self, results):
        proposals = results['proposals']
        if proposals.shape[1] not in (4, 5):
            raise AssertionError(
                'proposals should have shapes (n, 4) or (n, 5), '
                'but found {}'.format(proposals.shape))
        proposals = proposals[:, :4]

        if self.num_max_proposals is not None:
            proposals = proposals[:self.num_max_proposals]

        if len(proposals) == 0:
            proposals = np.array([0, 0, 0, 0], dtype=np.float32)
        results['proposals'] = proposals
        results['bbox_fields'].append('proposals')
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(num_max_proposals={})'.format(
            self.num_max_proposals)

