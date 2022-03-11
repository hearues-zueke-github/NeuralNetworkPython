#! /usr/bin/python2.7

import gzip
import os
import sys
import time

import cPickle as pkl
import numpy as np

from PIL import Image, ImageDraw

def print_variable(variable_name, globals_dict):
    print(variable_name+" = "+str(globals_dict[variable_name]))

def convert_list_to_image_14x14(l):
    "l should have a length 196 and value between 0 and 255"

    img = Image.new("RGB", (14, 14))
    pix = img.load()

    for y in xrange(0, 14):
        for x in xrange(0, 14):
            v = int(l[14*y+x]*255)
            pix[x, y] = (v, v, v)
        # for
    # for

    return img

def scale_image_array(array_2d, scale=1):
    h, w = array_2d.shape
    bigger_array = np.zeros((h*scale, w*scale))
    for j in xrange(0, h):
        for i in xrange(0, w):
            bigger_array[scale*j:scale*(j+1), scale*i:scale*(i+1)] = array_2d[j, i]
    return bigger_array

def convert_array_to_image(array):
    h, w = array.shape
    pix = np.zeros((h, w, 3))
    array = (array*255).astype(np.int)
    print("pix.shape: {}".format(pix.shape))
    pix[:] = array.reshape((h, w, 1))
    return Image.fromarray(pix.astype(np.uint8))

def concat_images_horizontal(img1, img2):
    x1, y1 = img1.size
    x2, y2 = img2.size

    img_new = Image.new("RGB", (x1+x2, y1))

    px, py, w, h = 0, 0, x1, y1
    box = (px, py, px+w, py+h)
    img_new.paste(img1, box)

    px, py, w, h = x1, 0, x2, y2
    box = (px, py, px+w, py+h)
    img_new.paste(img2, box)

    return img_new
# def concat_images_horizontal

def concat_images_vertical(img1, img2):
    x1, y1 = img1.size
    x2, y2 = img2.size

    img_new = Image.new("RGB", (x1, y1+y2))

    px, py, w, h = 0, 0, x1, y1
    box = (px, py, px+w, py+h)
    img_new.paste(img1, box)

    px, py, w, h = 0, y1, x2, y2
    box = (px, py, px+w, py+h)
    img_new.paste(img2, box)

    return img_new
# def concat_images_horizontal

get_abs_box = lambda x,y,w,h: (x,y,x+w,y+h)
def add_horizontal_strips(img, slice_width, strip_width, color):
    width, height = img.size

    slices = int(width/slice_width) - 1
    new_img = Image.new("RGB", (width+slices*strip_width, height))
    draw_new_img = ImageDraw.Draw(new_img)

    for i in xrange(0, slices):
        box1 = get_abs_box((slice_width+strip_width)*i, 0, slice_width, height)
        box2 = get_abs_box(slice_width*i, 0, slice_width, height)
        new_img.paste(img.crop(box2), box1)

        # if i < slices-1:
        draw_new_img.rectangle((get_abs_box((slice_width+strip_width)*i+slice_width, 0, strip_width, height)), fill=color)
    # for

    i = slices
    box1 = get_abs_box((slice_width+strip_width)*i, 0, slice_width, height)
    box2 = get_abs_box(slice_width*i, 0, slice_width, height)
    new_img.paste(img.crop(box2), box1)

    # print("size = "+str(new_img.size))
    return new_img
# def add_horizontal_strips

def add_vertical_strips(img, slice_height, strip_height, color):
    width, height = img.size

    slices = int(height/slice_height) - 1
    new_img = Image.new("RGB", (width, height+slices*strip_height))
    draw_new_img = ImageDraw.Draw(new_img)

    for i in xrange(0, slices):
        box1 = get_abs_box(0, (slice_height+strip_height)*i, width, slice_height)
        box2 = get_abs_box(0, slice_height*i, width, slice_height)
        new_img.paste(img.crop(box2), box1)

        # if i < slices-1:
        draw_new_img.rectangle((get_abs_box(0, (slice_height+strip_height)*i+slice_height, width, strip_height)), fill=color)
    # for

    i = slices
    box1 = get_abs_box(0, (slice_height+strip_height)*i, width, slice_height)
    box2 = get_abs_box(0, slice_height*i, width, slice_height)
    new_img.paste(img.crop(box2), box1)

    # print("size = "+str(new_img.size))
    return new_img
# def add_horizontal_strips

def save_pkl_file(obj, file_path):
    with gzip.open(file_path, "wb") as fout:
        pkl.dump(obj, fout)
    # with
# def save_pkl_file

def load_pkl_file(file_path):
    # Load the neural_network
    # print("Loading file "+file_path)

    with gzip.open(file_path, "rb") as fi:
        obj = pkl.load(fi)
    # with

    return obj
# def load_pkl_file

def check_create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

        return False

    return True

def test_time(f, a, k):
    start = datetime.now()
    f(*a, **k)
    end = datetime.now()
    return end - start
# def test_time

def test_time_args_direct(f, a, k):
    start = time.time()
    f(a, **k)
    end = time.time()
    return end - start
# def test_time_args_direct

def create_new_bws(bws_folder, nl):
    def get_random_bws(nl):
        return np.array([np.random.uniform(-1./np.sqrt(n), 1./np.sqrt(n), (m+1, n)) for m, n in zip(nl[:-1], nl[1:])])
    
    if not os.path.exists(bws_folder):
        os.makedirs(bws_folder)

    nl_str = "_".join(list(map(str, nl)))
    bws_path = bws_folder+"/bws_{}.npz".format(nl_str)
    if not os.path.exists(bws_path):
        bws = get_random_bws(nl)
        with open(bws_path, "wb") as f:
            np.savez_compressed(f, bws=bws)
    else:
        bws = np.load(bws_path)["bws"]

    return bws

def permutation_X_T(X, T):
    idxs = np.random.permutation(np.arange(0, X.shape[0]))
    return X[idxs], T[idxs]
