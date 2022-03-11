#! /usr/bin/python2.7

from __init__ import *
# import sys
# import os
# import gzip
# import timeit

# import cPickle as pkl
# import numpy as np
# import Utils as utils

# from copy import deepcopy
# from random import randint
# from DigitRecognition import main_digits_learning_testing_iterations
# from TrainedNetwork import TrainedNetwork
# from PIL import Image

def learn_autoencoder_network(iterations=1):
    main_digits_learning_testing_iterations("lnn_14x14_autoencoder_196_28_10_28_196.pkl.gz",
                                            "inputs_targets_autoencoder_14x14_train.pkl.gz",
                                            "inputs_targets_autoencoder_14x14_valid.pkl.gz",
                                            "lnn_14x14_autoencoder_196_28_10_28_196.pkl.gz",
                                            "lnn_14x14_autoencoder_statistics_196_28_10_28_196.txt",
                                            iterations,
                                            is_save_error=True)
# def learn_network_1

def get_output_picture():
    tn = TrainedNetwork()
    tn.set_file_path("lnn_14x14_autoencoder.pkl.gz")
    tn.load_network()
    nn = tn.nn

    # First calc forward many
    with gzip.GzipFile("autoencoder_14x14_train.pkl.gz", "rb") as fin:
        inputs, targets, values = pkl.load(fin)
    # end with

    length = 100*500

    bs = nn.biases
    ws = nn.weights
    outputs = nn.calculate_forward_many(inputs[:length], bs, ws)
    print("len(outputs) = "+str(len(outputs)))

    # Convert all outputs to lists and change values between 0 and 255 (0...1)
    outputs = [o.transpose().tolist()[0] for o in outputs]
    # print("outputs before = "+str(outputs))
    print("before changing")

    func = lambda x: 0.5 + (x - 0.4) * 1.25
    new_outputs = []
    for o in outputs:
        new_o = []
        for val in o:
            new_val = int(func(val) * 255)
            if new_val < 0:
                new_o.append(0)
            elif new_val > 255:
                new_o.append(255)
            else:
                new_o.append(new_val)
            # if
        # for
        new_outputs.append(new_o)
    # for
    outputs = new_outputs

    # print("outputs after = "+str(outputs))
    print("after changing")

    images = [utils.convert_list_to_image_14x14(l) for l in outputs]

    # for img, i in zip(images, xrange(0, len(images))):
    #     img.save("output_autoencoder_14x14/"+"output_train_nr_"+str(i).zfill(5)+".png")
    # # for

    # Merge all outputs together to one big file
    img_merge = Image.new("RGB", (14*100, 14*500))
    for y in xrange(0, 500):
        for x in xrange(0, 100):
            px, py, w, h = 14*x, 14*y, 14, 14
            box = (px, py, px+w, py+h)

            img_merge.paste(images[10*y+x], box)
        # for
    # for

    img_merge.save("output_autoencoder_14x14/merge_output.png")
# def get_output_picture

# create_raw_14x14_train_valid_test()
# create_autoencoder_inputs_targets_14x14()

# learn_autoencoder_network(1)

# for i in xrange(0, 10): print("Nr. "+str(i)); learn_autoencoder_network(5)
# change_weights_of_network()

# get_output_picture()
# test_image_creating()
