import cv2
import numpy as np
import os
import shutil
import pdb
from PIL import Image,ImageFont,ImageDraw

font_size = 24
pil_font = ImageFont.truetype("tools/simhei.ttf", font_size)

COLOR = {
        'green':(0,255,0),
        'blue':(255,0,0),
        'red':(0,0,255),
        'yellow':(0,210,255)
        }

def stack_image_with_blank(img_list, interval=20):
    new_list = []
    h, w, c = img_list[0].shape
    blank = np.zeros((h, interval, c))
    num_images = len(img_list)
    for idx in range(num_images - 1):
        new_list.append(img_list[idx])
        new_list.append(blank)
    new_list.append(img_list[-1])
    stacked_img = np.concatenate(new_list, 1)
    return stacked_img

def PIL_text(image, box, text, color):
    ''' Put text on cv2 image(ndarray) with PIL.
    '''
    pil_im = Image.fromarray(np.uint8(image))
    draw = ImageDraw.Draw(pil_im)
    x = int(box[0])
    y = int(box[1]-font_size-2)
    twidth, theight = draw.textsize(text, font=pil_font)
    #draw.rectangle(((x,y),(x+twidth,y+theight)), fill=color[::-1]) # text background color
    #draw.text((x,y), text, (255,255,255), font=pil_font)
    draw.text((x,y), text, color[::-1], font=pil_font)
    image = np.array(pil_im)
    return image

def read_img(file_path, opt, window=(-1400,300)):
    mode = opt.mode
    enlarge_factor = opt.enlarge_factor
    if mode == 'uint16':
        image = cv2.imread(file_path, -1)
        image = windowing(image, window)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif mode == 'uint8':
        image = cv2.imread(file_path, -1)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif mode == 'rgb':
        image = cv2.imread(file_path)
    image = cv2.resize(image, (0, 0), fx=enlarge_factor, fy=enlarge_factor)
    return image

def windowing(im, win=(-1024,1050)):
    # Scale intensity from win[0]~win[1] to float numbers in 0~255
    #assert im.dtype == np.uint16
    im1 = im.astype(float)
    im1 -= 32768
    im1 -= win[0]
    im1 /= win[1] - win[0]
    im1[im1 > 1] = 1
    im1[im1 < 0] = 0
    im1 *= 255
    return im1.astype(np.uint8)

def reset_dir(save_dir):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir,True)
    os.makedirs(save_dir)
