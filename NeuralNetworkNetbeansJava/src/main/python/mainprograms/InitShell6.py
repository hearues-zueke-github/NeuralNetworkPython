#! /usr/bin/python2.7

import sys
import os

import cPickle as pkl
import Utils as utils

from PIL import Image
from TrainedNetwork import TrainedNetwork
from DigitRecognition import get_picture_from_vector

# inputs, targets, values = load_pkl_file("inputs_targets_train_14x14.pkl.gz")

# # Calc forward
# inputs_a, targets_a, values_a = 

# # show_picture(inputs[0], values[0], 0, 14, 14, 10)

for i in xrange(1, 5):
    iterations = i*400
    nn1 = utils.load_pkl_file("autoencoder_backup_"+str(iterations)+"_iterations/lnn_14x14_autoencoder_196_28_10_28_196.pkl.gz")
    nn2 = utils.load_pkl_file("autoencoder_backup_"+str(iterations)+"_iterations/lnn_14x14_autoencoder_196_56_10_56_196.pkl.gz")
    nn3 = utils.load_pkl_file("autoencoder_backup_"+str(iterations)+"_iterations/lnn_14x14_autoencoder_196_56_28_10_28_56_196.pkl.gz")
    print("Loaded Network!")

    # First calc forward many
    inputs, targets, values = utils.load_pkl_file("inputs_targets_14x14_valid.pkl.gz")
    # inputs, targets, values = utils.load_pkl_file("inputs_targets_14x14_train.pkl.gz")
    print("Loaded Inputs, Targets and Values!")

    length = 100

    outputs1 = nn1.calculate_forward_many(inputs[:length], nn1.biases, nn1.weights)
    outputs2 = nn2.calculate_forward_many(inputs[:length], nn2.biases, nn2.weights)
    outputs3 = nn3.calculate_forward_many(inputs[:length], nn3.biases, nn3.weights)
    print("Calculated Outputs!")

    img_list = []
    for i in xrange(0, 4):
        img1 = get_picture_from_vector(inputs[i], values[i], 0, 14, 14, 10)
        img1_1 = get_picture_from_vector(outputs1[i], values[i], 0, 14, 14, 10)
        img2_1 = get_picture_from_vector(outputs2[i], values[i], 0, 14, 14, 10)
        img3_1 = get_picture_from_vector(outputs3[i], values[i], 0, 14, 14, 10)

        img_new = utils.concat_images_horizontal(img1, img1_1)
        img_new = utils.concat_images_horizontal(img_new, img2_1)
        img_new = utils.concat_images_horizontal(img_new, img3_1)
        img_list.append(img_new)
        # img_new.show()
    # for

    img_new_complete = img_list[0]
    for img_iter in img_list[1:]:
        img_new_complete = utils.concat_images_vertical(img_new_complete, img_iter)
    # for

    # img_new_complete.show()
    # print("size = "+str(img_new_complete.size))
    new_img_comp = utils.add_horizontal_strips(img_new_complete, 140, 5, (205, 52, 52))
    # new_img_comp.show()
    new_img_comp = utils.add_vertical_strips(new_img_comp, 140, 10, (85, 82, 186)) #(205, 52, 52))

    img_label = Image.open("pictures/auto_encoder_14x14_label.png")
    new_img_comp = utils.concat_images_vertical(img_label, new_img_comp)

    new_img_comp.show()
    new_img_comp.save("pictures/auto_encoder_14x14_"+str(iterations)+"_iterations.png")
# for

# img2_1 = get_picture_from_vector(inputs[0], values[0], 0, 14, 14, 10)
# img3_1 = get_picture_from_vector(inputs[0], values[0], 0, 14, 14, 10)

