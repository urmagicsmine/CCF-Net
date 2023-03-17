import numpy as np
from pycocotools.coco import COCO
from .registry import DATASETS
import json
from.coco import CocoDataset


@DATASETS.register_module
class TianchiDataset(CocoDataset):

    CLASSES = ('nodule', 'suotiao', 'dongmai', 'linba')

    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.coco.getImgIds()
        img_infos = []
        json_file = json.load(open(ann_file, 'rb'))
        images = json_file['images']

        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            for j in images:
                if j['id'] == i:
                    #info['slice_intv'] = j['slice_intv']
                    #if 'slice_intv' in j.keys():
                        #info['slice_intv'] = j['slice_intv']
                    #else:
                    # set 2.5 to avoid interpolating
                    # (or set SLICE_INTERVAL in pipelines to 5)
                    info['slice_intv'] = 2.5

                    break
            img_infos.append(info)
        return img_infos

    def get_all_gt_bboxes(self):
        all_gt_bboxes = []
        for i in range(len(self.img_infos)):
            ann = self.get_ann_info(i)
            all_gt_bboxes.append(ann['bboxes'])
        return all_gt_bboxes
