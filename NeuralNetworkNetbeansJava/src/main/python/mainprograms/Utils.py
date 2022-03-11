#! /usr/bin/python2.7

import gzip
import os
import time
import datetime

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
            v = l[14*y+x]
            pix[x, y] = (v, v, v)
        # for
    # for

    return img
# def convert_list_to_image_14x14

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

def test_time(f, a, k):
    start = datetime.datetime.now()
    f(*a, **k)
    end = datetime.datetime.now()
    return end - start

def test_time_with_return(f, a, k):
    start = datetime.datetime.now()
    ret = f(*a, **k)
    end = datetime.datetime.now()
    return (end - start, ret)

def test_time_args_direct(f, a, k):
    start = time.time()
    f(a, **k)
    end = time.time()
    return end - start

def get_n_to_2_logistic_samples(dimN, amount):
    inputs = np.zeros((0, dimN))
    targets = np.zeros((0, 2))

    sigmoid = lambda x: 1. / (1. + np.exp(-x))
    vecsig = np.vectorize(sigmoid)
    factors = np.random.randint(-5, 6, (2, dimN-1))
    # print("factors =\n{}".format(factors))

    for _ in xrange(amount):
        input = (np.random.random((dimN, ))-0.5)*2*3
        inputs = np.vstack((inputs, input))
        # print("input1 =\n{}".format(input))
        input = np.array([input[:-1], input[1:]])
        # print("input2 =\n{}".format(input))
        input = np.sum(input * factors, axis=1)
        # print("input3 after sum =\n{}".format(input))
        # print("target 0\n{}".format(vecsig(input)))
        targets = np.vstack((targets, vecsig(input)))

    return inputs, targets
