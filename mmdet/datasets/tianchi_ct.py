import os.path as osp

import mmcv
import numpy as np
from torch.utils.data import Dataset

from .pipelines import Compose
from .registry import DATASETS


@DATASETS.register_module
class TianchictDataset(Dataset):
    """Tianchi CT dataset for inference.
    """

    CLASSES = ('nodule', 'strip', 'dongmai', 'linba')

    def __init__(self,
                 sub_dir,
                 pipeline,
                 image_root=None,
                 mask_root=None,
                 test_mode=True,
                 slice_expand=10,
                 sub_dir_list=None):
        self.sub_dir = sub_dir
        self.image_root = image_root
        self.mask_root = mask_root
        self.test_mode = test_mode
        if sub_dir is None:
            with open(sub_dir_list, 'r') as f:
                lines = f.readlines()
            sub_dirs = [line.strip() for line in lines]
            sub_dir = sub_dirs[0]
        self.image_path = osp.join(self.image_root, sub_dir, 'norm_image.npz')
        self.image_tensor = np.load(open(self.image_path, 'rb'))['data']
        if mask_root is not None:
            self.mask_tensor = np.load(open(osp.join(self.image_root, sub_dir, 'mask_image.npz'), 'rb'))['data']
        else:
            self.mask_tensor = None

        self.img_infos = self.load_annotations(self.image_tensor, self.mask_tensor, slice_expand)
        # processing pipeline
        self.pipeline = Compose(pipeline)

    def __len__(self):
        return len(self.img_infos)

    def get_img_path(self):
        return self.image_path

    def load_annotations(self, image_tensor, mask_tensor, slice_expand):
        if mask_tensor is not None:
            if mask_tensor.dtype == np.bool:
                mask_tensor = np.uint8(mask_tensor) * 255
            mask_index = np.where(mask_tensor == 1)
            z_min, y_min, x_min = [np.min(idx) for idx in mask_index]
            z_max, y_max, x_max = [np.max(idx) for idx in mask_index]
            depth, height, width = image_tensor.shape

            z_start = max(0, z_min - slice_expand)
            # z_end = depth as we use range(z_start, z_end) later; fix 'depth - 1' bug.
            z_end = min(depth, z_max + slice_expand + 1)
        else:
            z_start = 0
            z_end = image_tensor.shape[0]
        img_infos = []
        for slice_index in range(z_start, z_end):
            img_info = {}
            img_info['slice_index'] = slice_index
            img_info['filename'] = slice_index
            img_infos.append(img_info)
        return img_infos

    def pre_pipeline(self, results):
        results['image_tensor'] = self.image_tensor
        results['bbox_fields'] = []
        results['mask_fields'] = []

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)

    def prepare_test_img(self, idx):
        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

