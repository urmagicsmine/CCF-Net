import cv2
import os
import os.path as osp
import numpy as np
SLICE_INTERVAL = 2.5

def load_multislice_gray_png_3DCE(imname, num_slice=9, slice_intv=5,\
        zflip=False, lung_mip=False, multi_view=False ,window=(-1024, 1050)):
    # Load single channel image for lesion detection
    #if multi_view:
        ## perform three kind of windowing to generate num_slice output.
        #num_slice /= 3
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
            # print('file not found:', img_name)
            delta -= np.sign(delta)
            img_name = '%03d.png' % (slice_idx + delta)
            full_path = os.path.join('./', *name_slice_list[:-1], img_name)
            if delta == 0:
                break
        return full_path

    def _load_data(img_name, delta=0):
        img_name = get_slice_name(img_name, delta)
        if img_name not in data_cache.keys():
            data_cache[img_name] = cv2.imread(img_name, -1)
            if data_cache[img_name] is None:
                print('file reading error:', img_name, os.path.exists(img_name))
                assert not data_cache[img_name] == None
        return data_cache[img_name]

    def _load_multi_data(im_cur, imname, num_slice, slice_intv, zfilp=False):
        ims = [im_cur]
        # find neighboring slices of im_cur
        rel_pos = float(SLICE_INTERVAL) / slice_intv
        a = rel_pos - np.floor(rel_pos)
        b = np.ceil(rel_pos) - rel_pos
        sequence_flag = True
        if zflip:
            if np.random.rand() > 0.5:
                sequence_flag = False
        if sequence_flag:  #TODO: This is hard to read. Needs fix
            if a == 0:  # required SLICE_INTV is a divisible to the actual slice_intv, don't need interpolation
                for p in range((num_slice - 1) // 2):
                    im_prev = _load_data(imname, - rel_pos * (p + 1))
                    im_next = _load_data(imname, rel_pos * (p + 1))
                    ims = [im_prev] + ims + [im_next]
                # when num_slice is even number,got len(ims) with num_slice-1. Add 1 slice.
                if num_slice % 2 == 0:
                    im_next = _load_data(imname, rel_pos * (p + 2))
                    ims += [im_next]
            else:
                for p in range((num_slice - 1) // 2):
                    intv1 = rel_pos * (p + 1)
                    slice1 = _load_data(imname, - np.ceil(intv1))
                    slice2 = _load_data(imname, - np.floor(intv1))
                    im_prev = a * slice1 + b * slice2  # linear interpolation
    
                    slice1 = _load_data(imname, np.ceil(intv1))
                    slice2 = _load_data(imname, np.floor(intv1))
                    im_next = a * slice1 + b * slice2
                    ims = [im_prev] + ims + [im_next]
                # when num_slice is even number,got len(ims) with num_slice-1. Add 1 slice.
                if num_slice % 2 == 0:
                    intv1 = rel_pos * (p + 2)
                    slice1 = _load_data(imname, np.ceil(intv1))
                    slice2 = _load_data(imname, np.floor(intv1))
                    im_next = a * slice1 + b * slice2
                    ims += [im_next]
        else:
            if a == 0:  # required SLICE_INTV is a divisible to the actual slice_intv, don't need interpolation
                for p in range((num_slice - 1) // 2):
                    im_next = _load_data(imname, - rel_pos * (p + 1))
                    im_prev = _load_data(imname, rel_pos * (p + 1))
                    ims = [im_prev] + ims + [im_next]
            else:
                for p in range((num_slice - 1) // 2):
                    intv1 = rel_pos * (p + 1)
                    slice1 = _load_data(imname, - np.ceil(intv1))
                    slice2 = _load_data(imname, - np.floor(intv1))
                    im_next = a * slice1 + b * slice2  # linear interpolation

                    slice1 = _load_data(imname, np.ceil(intv1))
                    slice2 = _load_data(imname, np.floor(intv1))
                    im_prev = a * slice1 + b * slice2
                    ims = [im_prev] + ims + [im_next]

        return ims

    data_cache = {}
    im_cur = cv2.imread(imname, -1)
    num_slice = num_slice
    ims = _load_multi_data(im_cur, imname, num_slice, slice_intv, zflip)
    #if len(ims) == 1:
        #ims = [ims[0], ims[0], ims[0]]
    ims = [im.astype(float) for im in ims]
    # Support MIP MinIP as auxiliary channel.
    if lung_mip:
        mig_im = np.max(cv2.merge(ims), axis = 2)
        min_im = np.min(cv2.merge(ims), axis = 2)
        center_im = ims[int((num_slice - 1)/2)]
        #im = np.vstack((mig_im, center_im, min_im))
        im = cv2.merge([mig_im, center_im, min_im])
    im = cv2.merge(ims)
    im = im.astype(np.float32, copy=False) - 32768
    #im = windowing(im, [-1024, 3071])
    if multi_view:
        im = multi_windowing(im)
    else:
        im = windowing(im, window)
    ### vis save ######
    #print(im.shape)
    #h, w, c = im.shape
    #vis_im = im[:,:,c//2].astype(np.uint8)
    #k = len(os.listdir('ori_img'))
    #cv2.imwrite('ori_img/%d.jpg' % k, vis_im)
    ###################
    return im

def windowing(im, win):
    # Scale intensity from win[0]~win[1] to float numbers in 0~255
    im1 = im.astype(np.float32)
    im1 -= win[0]
    im1 /= win[1] - win[0]
    im1[im1 > 1] = 1
    im1[im1 < 0] = 0
    im1 *= 255
    return im1

def multi_windowing(im):
    windows = [[-174,274],[-1493,484],[-534,1425]]
    #windows = [[-1493,484],[-200,300],[-174,274]]
    #windows = [[-1000,600],[-350,450],[-174,274]]
    #windows = [[-1400,200],[-135,215],[-174,274]]
    #windows = cfg.WINDOWING
    #assert im.shape[2] == 9,'im.shape != 9.'
    #assert c%3==0,'channel cannot devided by 3'
    #if cfg.LESION.NUM_IMAGES_3DCE == 2:
        #im_win1 = windowing(im, windows[0])
        #im_win2 = windowing(im, windows[1])
        #im = np.concatenate((im_win1, im_win2),axis=2)
    #else:
    #if im.shape[2] == 3:
    if True:
        im_win1 = windowing(im, windows[0])
        im_win2 = windowing(im, windows[1])
        im_win3 = windowing(im, windows[2])
        im = np.concatenate((im_win1, im_win2, im_win3),axis=2)
    #im_win1 = windowing(im, windows[0])
    #im_win2 = windowing(im, windows[1])
    #im = np.concatenate((im_win1, im_win2),axis=2)
    return im.astype(np.uint8)


