import cv2
import os
import os.path as osp
import numpy as np

def load_multislice_gray_png(imname, num_slice=3, zflip=False, lung_mip=False, extra_transform=None, zaug=False):
    # Load single channel image for general lung detection
    def get_slice_name(img_path, delta=0):
        if delta == 0:
            return img_path
        delta = int(delta)
        name_slice_list = img_path.split(os.sep)
        slice_idx = int(name_slice_list[-1][:-4])
        img_name = '%03d.png' % (slice_idx + delta)
        full_path = os.path.join('./', *name_slice_list[:-1], img_name)

        # if the slice is not in the dataset, use its neighboring slice
        while not os.path.exists(full_path):
            #print('file not found:', img_name)
            delta -= np.sign(delta)
            img_name = '%03d.png' % (slice_idx + delta)
            full_path = os.path.join('./', *name_slice_list[:-1], img_name)
            if delta == 0:
                break
        return full_path

    def _load_data(img_name, delta=0):
        img_name = get_slice_name(img_name, delta)
        if img_name not in data_cache.keys():
            data_cache[img_name] = cv2.imread(img_name, 0)
            if data_cache[img_name] is None:
                print('file reading error:', img_name, os.path.exists(img_name))
                assert not data_cache[img_name] == None
        return data_cache[img_name]


    def _load_multi_data(im_cur, imname, num_slice, zfilp=False, zaug=False):
        ims = [im_cur]
        # find neighboring slices of im_cure
        rel_pos = 1
        if zaug:
            if np.random.rand() > 0.5:
                rel_pos = 2
        sequence_flag = True
        if zflip:
            if np.random.rand() > 0.5:
                sequence_flag = False
        if sequence_flag:
            for p in range((num_slice-1)//2):
                im_prev = _load_data(imname, - rel_pos * (p + 1))
                im_next = _load_data(imname, rel_pos * (p + 1))
                ims = [im_prev] + ims + [im_next]
            #when num_slice is even number,got len(ims) with num_slice-1. Add 1 slice.
            if num_slice%2 == 0:
                im_next = _load_data(imname, rel_pos * (p + 2))
                ims = ims + [im_next]
        else:
            for p in range((num_slice-1)//2):
                im_next = _load_data(imname, - rel_pos * (p + 1))
                im_prev = _load_data(imname, rel_pos * (p + 1))
                ims = [im_prev] + ims + [im_next]
            #when num_slice is even number,got len(ims) with num_slice-1. Add 1 slice.
            if num_slice%2 == 0:
                im_prev = _load_data(imname, rel_pos * (p + 2))
                ims = [im_prev] + ims
        return ims
    # if use extra_transform, merge 3 thin slices as one thick slice
    if extra_transform == 'pseudo_thick':
        num_slice *= 3
    data_cache = {}
    im_cur = cv2.imread(imname, 0)
    ims = _load_multi_data(im_cur, imname, num_slice, zflip, zaug)
    #ims = [im.astype(float) for im in ims]
    # Support MIP MinIP as auxiliary channel.
    if lung_mip:
        mig_im = np.max(cv2.merge(ims), axis = 2)
        min_im = np.min(cv2.merge(ims), axis = 2)
        center_im = ims[int((num_slice - 1)/2)]
        #im = np.vstack((mig_im, center_im, min_im))
        im = cv2.merge([mig_im, center_im, min_im])
    elif extra_transform=='pseudo_thick':
        ims = cv2.merge(ims)
        h,w,c = ims.shape
        ims = ims.reshape(h, w, -1, 3)
        ims = np.mean(ims, axis=-1)
        im = ims.astype(np.uint8)
    elif extra_transform=='single_slice_as_rgb':
        center = len(ims) // 2
        ims = [ims[center], ims[center], ims[center]]
        im = cv2.merge(ims)
    elif extra_transform=='single_slice_thick_as_rgb':
        center = len(ims) // 2
        ims = [ims[center-1], ims[center], ims[center+1]] * 3
        ims = cv2.merge(ims)
        h,w,c = ims.shape
        ims = ims.reshape(h, w, -1, 3)
        ims = np.mean(ims, axis=-1)
        im = ims.astype(np.uint8)
    elif extra_transform==None:
        im = cv2.merge(ims)
    else:
        print('Not impl transform:', extra_transform)
    return im

def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))

def load_data(image_tensor, center_idx, delta):
    cur_idx = int(center_idx + delta)
    depth,_,_ = image_tensor.shape
    cur_idx = clamp(cur_idx, 0, depth-1)
    return image_tensor[cur_idx, :, :]

