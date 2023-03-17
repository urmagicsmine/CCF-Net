import argparse
import copy
import os
import os.path as osp
import numpy as np
import pdb
import cv2
import mmcv
import torch
from tools.eval_utils import COLOR, PIL_text, read_img, reset_dir, windowing, stack_image_with_blank
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.core.evaluation import eval_map, hit_miss, get_cls_results
import pycocotools.mask as mutils

REPLACE_CLASS_NAME = {
        'nodule': 'nodule',
        'suotiao': 'stripe',
        'linba': 'lymph',
        'dongmai': 'artery'}
MASK_COLOR = np.array((2, 166, 101)).astype(np.uint8)

class DetInfo(object):
    def __init__(self,
                file_path,
                file_name,
                det_bboxes,
                gt_bboxes,
                gt_labels,
                gt_masks):
        self.file_path= file_path
        self.file_name= file_name
        self.det_bboxes = det_bboxes
        self.gt_bboxes = gt_bboxes
        self.gt_masks = gt_masks
        self.hit = []
        self.miss = []
        self.tp = []
        self.fp = []
    def filter_bboxes(self, preserve_class, thresh):
        def filter_(obj, preserve_class):
            for i in range(len(self.det_bboxes)):
                if i == preserve_class:
                    continue
                obj[i] = np.array([])
        if len(self.det_bboxes[preserve_class])==0:
            return True
        elif max(self.det_bboxes[preserve_class][:,-1]) < thresh:
            return True
        filter_(self.hit, preserve_class)
        filter_(self.miss, preserve_class)
        filter_(self.tp, preserve_class)
        filter_(self.fp, preserve_class)
        return False

class DatasetEvaluator(object):
    def __init__(self, opt):
        self.opt = opt
        self.cfg = mmcv.Config.fromfile(opt.config)
        self.cfg.data.test.test_mode = True
        self.dataset = build_dataset(self.cfg.data.test)
        self.num_classes = len(self.dataset.cat_ids)
        self.data_path = self.cfg.data.test['img_prefix']
        self.get_gt_bboxes()

    def evaluate_bboxes(self, det_results):
        ''' Eval and add eval results to each image item.
        '''
        print('evaluating bboxes...')
        det_bboxes = [det_result.det_bboxes for det_result in det_results]
        for i in range(self.num_classes):
            cls_dets, cls_gts, cls_gt_ignore = get_cls_results(
                self.det_bboxes, self.gt_bboxes, self.gt_labels, None, i)
            for j in range(len(cls_dets)): # traverse each det_result
                tp, hit = hit_miss(cls_dets[j], cls_gts[j], None, iou_thr=0.5)
                det_results[j].tp.append(cls_dets[j][tp==1])
                det_results[j].fp.append(cls_dets[j][tp==0])
                det_results[j].hit.append(cls_gts[j][hit==1])
                det_results[j].miss.append(cls_gts[j][hit==0])
        return

    def load_det_results(self, result_file):
        ''' Load detection results and generate objects of class DetInfo.
        '''
        print('res file', result_file)
        if mmcv.is_str(result_file):
            assert result_file.endswith('.pkl')
            det_bboxes = mmcv.load(result_file)
        self.det_bboxes = det_bboxes
        assert len(det_bboxes) == len(self.gt_bboxes), \
            str(len(self.det_bboxes)) + 'vs' + str(len(self.gt_bboxes))
        det_results = []
        for i in range(len(det_bboxes)):
            det_results.append(
                    DetInfo(
                        self.data_path,
                        self.dataset.img_infos[i]['filename'],
                        det_bboxes[i],
                        self.gt_bboxes[i],
                        self.gt_labels[i],
                        self.gt_masks[i]))
        self.evaluate_bboxes(det_results)
        return det_results

    def get_gt_bboxes(self):
        ''' Load ground truth boxes.
        '''
        gt_bboxes = []
        gt_labels = []
        gt_masks = []
        for i in range(len(self.dataset)):
            ann = self.dataset.get_ann_info(i)
            #bboxes = np.hstack((ann['bboxes'], ann['labels'].reshape(-1, 1)))
            bboxes = ann['bboxes']
            labels = ann['labels']
            masks = ann['masks'] if 'masks' in ann.keys() else None
            gt_bboxes.append(bboxes)
            gt_labels.append(labels)
            gt_masks.append(masks)
        self.gt_bboxes = gt_bboxes
        self.gt_labels = gt_labels
        self.gt_masks = gt_masks

    def eval_map(self, det_results):
        ''' Call 'eval_map' function impl by mmdetection.
            This map is not the same as coco map.
        '''
        det_bboxes = [det.det_bboxes for det in det_results]
        mean_ap, eval_results = eval_map(det_bboxes, self.gt_bboxes, self.gt_labels)

    def visualize(self, det_results, det_result_others, draw_class=None):
        ''' Visualization func, draw results for each image in coco val set.
            Args:
                det_results: DetInfo objects.
                det_result_list: Another detect results for comparation.
                draw_class:  Indicate current class, if visualize each class separately.
        '''
        print('start drawing...')
        count = 0
        reset_dir(self.opt.save_dir)
        for idx, det_item in enumerate(det_results):
            det = copy.deepcopy(det_item)
            if not draw_class==None:   # skip this image if there are no detection results
                is_empty= det.filter_bboxes(int(draw_class), self.opt.thresh)
                if is_empty:
                    continue
            img_path = os.path.join(det.file_path, det.file_name)
            origin_image = read_img(img_path, self.opt)
            if self.opt.method == 'tpfp':
                image, flag = self.draw_box(origin_image, self.opt,
                        [det.tp, det.fp],
                        [COLOR['green'], COLOR['red']])
            elif self.opt.method == 'fp':
                image, _ = self.draw_box(origin_image, self.opt,
                        [det.hit, det.miss],
                        [COLOR['green'], COLOR['blue']])
                image, flag = self.draw_box(origin_image, self.opt,
                        [det.fp],
                        [COLOR['red']])
            elif self.opt.method == 'gt':
                image, flag = self.draw_box(origin_image, self.opt,
                        [det.hit, det.miss],
                        [COLOR['green'], COLOR['red']])
            elif self.opt.method == 'all':
                image, flag = self.draw_box(origin_image, self.opt,
                        [det.tp, det.fp, det.hit, det.miss],
                        [COLOR['green'], COLOR['red'], COLOR['green'], COLOR['red']])
            elif self.opt.method == 'mask':
                image, _ = self.draw_box(origin_image, self.opt,
                        [det.hit, det.miss],
                        [COLOR['green'], COLOR['red']])
                image, flag = self.draw_mask(image, det.gt_masks)
                image = np.vstack((origin_image, image))
            elif self.opt.method == 'compare':
                image, flag = self.draw_box(origin_image, self.opt,
                        [det.tp, det.fp],
                        [COLOR['green'], COLOR['red']])
                # this part is used for choose proper visualizaion images.
                #if not draw_class==None:
                    #det_compare = copy.deepcopy(det_results_cpr[idx])
                    #is_empty = det_compare.filter_bboxes(int(draw_class), self.opt.thresh)
                    #print('is_empty', is_empty)
                    ##if len(det.tp[draw_class]) <= len(det_compare.tp[draw_class]):
                        ##continue
                #else:
                    #pass
                    #if len(np.vstack(det.tp)) <= len(np.vstack(det_compare.tp)):
                        #continue
                gt_image, _ = self.draw_box(origin_image, self.opt,
                            [det.hit, det.miss],
                            [COLOR['green'], COLOR['green']])
                # draw for the first comperasion detection results.
                image_cpr_list = []
                for det_other in det_result_others:
                    image_cpr, _ = self.draw_box(origin_image, self.opt,
                        [det_other[idx].tp, det_other[idx].fp],
                        [COLOR['green'], COLOR['red']])
                    image_cpr_list.append(image_cpr)
                #image = np.hstack((gt_image, *image_cpr_list, image))
                image = stack_image_with_blank([gt_image, *image_cpr_list, image])

            if flag:
                file_name = det.file_name.replace('/','_') \
                        if '/' in det.file_name else det.file_name
                save_path = os.path.join(self.opt.save_dir, file_name)
                save_flag = cv2.imwrite(save_path, image)
                count += 1
        print(' draw :%d / %d' %(count, len(det_results)))

    def draw_box(self, img, class_name, all_boxes, colors):
        ''' draw bboxes and ground truths on one image.
        '''
        image = img.copy()
        draw_flag = False
        for boxes_of_one_task, color in zip(all_boxes, colors):
            for boxes, class_name in zip(boxes_of_one_task, self.dataset.CLASSES):
                # replace as standard name for Tianchi Dataset
                if class_name in REPLACE_CLASS_NAME:
                    class_name = REPLACE_CLASS_NAME[class_name]
                for box in boxes:
                    if len(box)==4:
                        text = class_name
                    elif len(box)==5:
                        if box[4] < self.opt.thresh:
                            continue
                        #text = class_name+ ' ' + str(box[4])[:4] # draw classname and score
                        text = str(box[4])[:4]                    # draw score only
                    box[:4] = box[:4] * self.opt.enlarge_factor
                    cv2.rectangle(image, (int(box[0]), int(box[1])), \
                            (int(box[2]), int(box[3])), color, 2)
                    image = PIL_text(image, box, text, color[::-1])
                    draw_flag = True
        return image, draw_flag

    def draw_mask(self, img, mask_encoded):
        image = img.copy()
        mask = mutils.decode(mask_encoded)
        if mask.shape[2] != 1:
            mask = np.sum(mask, axis=2)
        mask = np.dstack((mask, mask, mask))
        mask *= MASK_COLOR
        image[mask>0] = image[mask>0]*0.5 + mask[mask>0]*0.5
        image = image.astype(np.uint8)
        return image, True



def parse_args():
    parser = argparse.ArgumentParser(description='Coco Detection Analyze Tools')
    parser.add_argument('--config', help='Test config file path')
    parser.add_argument('--dataset', help='choose from [train, test]', default='test')
    parser.add_argument('--json_out', help='Output result file name with pkl extension', type=str)
    parser.add_argument('--json_out_others', help='Another output result file name with pkl extension', type=str, default=None, nargs='+')
    parser.add_argument('--method', help='Drawing method', default='all', type=str)
    parser.add_argument('--mode', help='data type of image, rgb for coco, uint8 fot LGD, uint16 for Deeplesion & Tianchi', default='rgb', type=str,
            choices=['rgb', 'uint8', 'uint16'])
    parser.add_argument('--thresh', help='Score threshold for visualization', default=0.05, type=float)
    parser.add_argument('--enlarge_factor', help='Factor to enlarge visualization result images', default=1.0, type=float)
    parser.add_argument('--draw_class', help='Specify one class(int) for visualization. \
                    Keep default(None) to visualize all class', default=None)
    parser.add_argument('--class_separate', help='Set to draw all classes on one image.', action='store_true')
    parser.add_argument('--save_dir', help='path to save visualization results', default='logs/visualize')
    args = parser.parse_args()
    return args


def main():

    # 1. Init and load results
    args = parse_args()
    evaluator = DatasetEvaluator(args)
    print(args.json_out_others)
    det_results = evaluator.load_det_results(args.json_out)
    if args.json_out_others:
        det_results_others = [evaluator.load_det_results(item) for item in args.json_out_others]
    save_dir = args.save_dir

    # 2. Visualization
    if args.class_separate:
        for idx, class_name in enumerate(evaluator.dataset.CLASSES):
            # save visualize results of different classes seperately
            args.save_dir = os.path.join(save_dir, class_name)
            evaluator.visualize(det_results, det_results_others, draw_class=idx)
    else:
        evaluator.visualize(det_results, det_results_others)

if __name__ == '__main__':
    main()


