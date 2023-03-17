import argparse
import os
import os.path as osp
import shutil
import tempfile
import numpy as np

import mmcv
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel, DataContainer
from mmcv.runner import get_dist_info, load_checkpoint

from mmdet.apis import init_dist
from mmdet.core import coco_eval, results2json, wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset, build_ct_dataset
from mmdet.datasets.pipelines.image_io import load_data
from mmdet.models import build_detector
from mmdet.datasets.pipelines import Compose

def ori_single_gpu_test(model, data_loader, show=False):
    model.eval()

    results_dict = {}
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        results_dict[data['img_meta'][0].data[0][0]['filename']] = result
        #results.append(result)

        if show:
            model.module.show_result(data, result)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results_dict

def get_input_dict(image_tensor, slice_index):
    results = {}
    img_info = {}
    img_info['slice_index'] = slice_index
    img_info['filename'] = slice_index
    results['img_info'] = img_info
    results['image_tensor'] = image_tensor
    return results

def get_range(image_tensor, mask_tensor, slice_expand):
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
    return z_start, z_end


def single_gpu_test(model, sub_dir_list, image_root, mask_root, test_pipeline, slice_expand, data_cmp, show=False):
    model.eval()
    prog_bar = mmcv.ProgressBar(len(sub_dir_list))

    all_results_dicts = {}
    for i, sub_dir in enumerate(sub_dir_list):
        #with torch.no_grad():
        if True:
            image_path = osp.join(image_root, sub_dir, 'norm_image.npz')
            image_tensor = np.load(open(image_path, 'rb'))['data']
            if mask_root is not None:
                mask_path = osp.join(mask_root, sub_dir, 'mask_image.npz')
                mask_tensor = np.load(open(mask_path, 'rb'))['data']
            else:
                mask_tensor = None
            z_start, z_end = get_range(image_tensor, mask_tensor, slice_expand)
            results_dict = {}
            for slice_index in range(z_start, z_end):
                data_input = get_input_dict(image_tensor, slice_index)
                data = test_pipeline(data_input)
                # Convert dataset item to dataloader item. Specifically, Datacontainer's data add []
                # And image tensor add a newaxis as batchsize
                data['img_meta'] = [DataContainer([[data['img_meta'][0].data]], cpu_only=True)]
                data['img'][0] = torch.unsqueeze(data['img'][0], 0)
                with torch.no_grad():
                    result = model(return_loss=False, rescale=not show, **data)
                results_dict[slice_index] = [None, None, result]
        all_results_dicts[image_path] = results_dict
        #results.append(result)

        if show:
            model.module.show_result(data, result)

        prog_bar.update()
    return all_results_dicts


def multi_gpu_test(model, data_loader, tmpdir=None):
    model.eval()
    results_dict = {}
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results_dict[data['img_meta'][0].data[0][0]['filename']] = result

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    results = collect_results(results, len(dataset), tmpdir)

    return results_dict


def collect_results(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--json_out',
        help='output result file name without extension',
        type=str)
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    assert args.out or args.show or args.json_out, \
        ('Please specify at least one operation (save or show the results) '
         'with the argument "--out" or "--show" or "--json_out"')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    if args.json_out is not None and args.json_out.endswith('.json'):
        args.json_out = args.json_out[:-5]

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test_ct_all.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_ct_dataset(cfg.data.test_ct_all)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    data = next(iter(data_loader))

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES


    sub_dir_path = cfg.data.test_ct_all.sub_dir_list
    with open(sub_dir_path) as f:
        lines = f.readlines()
        sub_dir_list = [line.strip() for line in lines]

    model = MMDataParallel(model, device_ids=[0])
    outputs = single_gpu_test(model,
                    sub_dir_list,
                    cfg.data.test_ct_all.image_root,
                    cfg.data.test_ct_all.mask_root,
                    Compose(cfg.data.test_ct_all.pipeline),
                    cfg.data.test_ct_all.slice_expand,
                    data,
                    show=False)

    """
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show)
    else:
        model = MMDistributedDataParallel(model.cuda())
        outputs = multi_gpu_test(model, data_loader, args.tmpdir)
    """
    #model = MMDataParallel(model, device_ids=[0])
    #outputs = single_gpu_test(model, data_loader, args.show)

    rank, _ = get_dist_info()
    if args.out and rank == 0:
        print('\nwriting results to {}'.format(args.out))
        mmcv.dump(outputs, args.out)


if __name__ == '__main__':
    main()
