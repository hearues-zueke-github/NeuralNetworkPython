#! /usr/bin/python2.7

from __init__ import *
# import numpy as np

# import DigitRecognition as digitrecognition
# from PIL import Image

def test_smaller_factor_set(smaller_factor):
    print("before:\nsmaller_factor = "+str(smaller_factor))

    dict_factors = [1., 2., 4.]
    for df1, df2 in zip(dict_factors[:-1], dict_factors[1:]):
        if smaller_factor - df1 < (df2 - df1) / 2.:
            smaller_factor = df1
            break
        # else:
            # smaller_factor = df2
        # if
    # for
    if smaller_factor > dict_factors[-1]:
        smaller_factor = dict_factors[-1]
    # if
    
    print("after:\nsmaller_factor = "+str(smaller_factor))
# def test_smaller_factor_set

def get_train_set_picture():
    inputs_targets_train = digitrecognition.load_pkl_file("inputs_targets_train_14x14.pkl.gz")

    inputs = inputs_targets_train[0]
    values = inputs_targets_train[2]

    wi = 100
    hi = 500

    w = 14
    h = 14

    wmax = wi * w
    hmax = hi * h

    data_whole = np.zeros((hmax, wmax, 3), dtype=np.uint8)
    data_string = ""

    for y in xrange(0, hi):
        for x in xrange(0, wi):
            temp_1d = inputs[y*wi + x].transpose().tolist()[0]
            temp_2d = [temp_1d[i*14:(i+1)*14] for i in xrange(0, 14)]

            for ys in xrange(0, 14):
                for xs in xrange(0, 14):
                    color_value = int(float(temp_2d[ys][xs]) * 255.)
                    data_whole[y*14+ys, x*14+xs] = (color_value, color_value, color_value)
                # for
            # for

            data_string += str(values[y*wi + x])
        # for
        data_string += "\n"
        print("layer "+str(y)+" finished!")
    # for

    img = Image.fromarray(data_whole, "RGB")
    img.save("inp_targ_train_14x14_whole.png")

    with open("inp_targ_train_14x14_whole_values.txt", "w") as fo:
        fo.write(data_string)
    # with
# def get_train_set_picture

def get_valid_set_picture():
    inputs_targets_valid = digitrecognition.load_pkl_file("inputs_targets_valid_14x14.pkl.gz")

    inputs = inputs_targets_valid[0]
    values = inputs_targets_valid[2]

    wi = 100
    hi = 100

    w = 14
    h = 14

    wmax = wi * w
    hmax = hi * h

    data_whole = np.zeros((hmax, wmax, 3), dtype=np.uint8)
    data_string = ""

    for y in xrange(0, hi):
        for x in xrange(0, wi):
            temp_1d = inputs[y*wi + x].transpose().tolist()[0]
            temp_2d = [temp_1d[i*14:(i+1)*14] for i in xrange(0, 14)]

            for ys in xrange(0, 14):
                for xs in xrange(0, 14):
                    color_value = int(float(temp_2d[ys][xs]) * 255.)
                    data_whole[y*14+ys, x*14+xs] = (color_value, color_value, color_value)
                # for
            # for

            data_string += str(values[y*wi + x])
        # for
        data_string += "\n"
        print("layer "+str(y)+" finished!")
    # for

    img = Image.fromarray(data_whole, "RGB")
    img.save("inp_targ_valid_14x14_whole.png")

    with open("inp_targ_valid_14x14_whole_values.txt", "w") as fo:
        fo.write(data_string)
    # with
# def get_train_set_picture
