from argparse import ArgumentParser 
import argparse
import os
import os.path as osp
import shutil
import tempfile
import json
import mmcv
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, load_checkpoint

from mmdet.apis import init_dist
from mmdet.core import coco_eval, results2json, wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from pycocotools.coco import COCO
from eval_tools.eval_FROC import eval_FROC


def main():
    parser = ArgumentParser(description='COCO Error Analysis Tool')
    parser.add_argument('result', help='result file (json format) path')
    parser.add_argument(
        '--config',
        default='./configs/mia_configs/faster_rcnn_hybrid_p3d18ba_res34_3dpfn9_as16_minms512_lesion_zflip_freeze0_fp16_optrange_fromcoco_2x.py',
        help='annotation file path')
    parser.add_argument(
        '--types', type=str, nargs='+', default=['bbox'], help='result types')
    args = parser.parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    outputs = mmcv.load(args.result)
    if True:
        avgFP = [0.5, 1, 2, 3, 4, 8]
        iou_th = 0.5
        all_boxes = [None]
        for cls in range(len(outputs[0])):
            tmp = [i[cls] for i in outputs]
            all_boxes.append(tmp)
        result_files = results2json(dataset, outputs, './')
        class_num = len(dataset.CLASSES)
        eval_FROC(result_files, ['bbox'], class_num, dataset.coco, all_boxes, avgFP, iou_th)

if __name__ == '__main__':
    main()

