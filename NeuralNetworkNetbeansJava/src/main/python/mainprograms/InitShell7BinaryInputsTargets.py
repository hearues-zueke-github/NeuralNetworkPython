#! /usr/bin/python2.7

import sys
import os

import cPickle as pkl
import Utils as utils
import BinaryAdder as binadd
import UtilsNeuralNetwork as utilsnn

from TrainedNetwork import TrainedNetwork
from DigitRecognition import get_picture_from_vector

binary_adder_sizes = [2, 3, 4, 5, 6, 8]
list_of_neural_list = [[x*2, x*2-1, x+1] for x in binary_adder_sizes]

print("list_of_neural_list = "+str(list_of_neural_list))

for neural_list in list_of_neural_list:
    utilsnn.create_new_random_network(neural_list, "_binadder")

for i in binary_adder_sizes:
    inputs_targets = binadd.get_binaryadder_inputs_targets(i)
    file_path = "inputs_targets_binary_"+str(i)+"_bits.pkl.gz"

    utils.save_pkl_file(inputs_targets, file_path)
