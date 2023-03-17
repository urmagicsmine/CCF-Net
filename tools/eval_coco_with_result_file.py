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
import pdb

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('--json_file', help='json file(*.bbox.json)')
    parser.add_argument('--pkl_file', help='pkl file (*.pkl)')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--eval_froc', action='store_true', help='show froc results')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    #eval_types = [args.eval]
    eval_types = ['bbox']
    cfg = mmcv.Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.test)

    result_files = args.json_file
    #filter_size = [0, 32]
    filter_size = [32, 400]
    print('\n### Evaluating AP with all results')
    coco_eval(result_files, eval_types, dataset.coco, filter_size=None)
    print('\n### Evaluating AP with ignoring bbox whose size in', filter_size)
    coco_eval(result_files, eval_types, dataset.coco, filter_size=filter_size)

    if args.eval_froc:
        outputs = mmcv.load(args.pkl_file)
        avgFP = [0.5, 1, 2, 3, 4, 8, 16, 32, 64]
        iou_th = 0.5
        all_boxes = [None]
        for cls in range(len(outputs[0])):
            tmp = [i[cls] for i in outputs]
            all_boxes.append(tmp)
        #pdb.set_trace()
        eval_FROC(dataset, all_boxes, avgFP, iou_th)
        eval_FROC(dataset, all_boxes, avgFP, iou_th, size_filter=filter_size)


if __name__ == '__main__':
    main()

