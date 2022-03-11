#! /usr/bin/python2.7

from __init__ import *
# from NeuralNetworkDecimalMultiprocess import NeuralNetwork
# from TrainedNetwork import TrainedNetwork
# from DigitRecognition import main_digits_learning_testing_iterations
# from Utils import load_pkl_file, \
#                   save_pkl_file

# import os

def learn_14x14_196_28_10_28_196_autoencoder_network(iterations=1):
    main_digits_learning_testing_iterations("lnn_14x14_autoencoder_196_28_10_28_196.pkl.gz",
                                            "inputs_targets_autoencoder_14x14_train.pkl.gz",
                                            "inputs_targets_autoencoder_14x14_valid.pkl.gz",
                                            "lnn_14x14_autoencoder_196_28_10_28_196.pkl.gz",
                                            "lnn_14x14_autoencoder_statistics_196_28_10_28_196.txt",
                                            iterations,
                                            is_save_error=True)

def learn_14x14_196_98_10(iterations=1):
    main_digits_learning_testing_iterations("lnn_14x14_196_98_10.pkl.gz",
                                            "inputs_targets_train_14x14.pkl.gz",
                                            "inputs_targets_valid_14x14.pkl.gz",
                                            "lnn_14x14_196_98_10.pkl.gz",
                                            "statistics_14x14_196_98_10.txt",
                                            iterations)

def learn_14x14_196_56_28_10(iterations=1):
    main_digits_learning_testing_iterations("lnn_14x14_196_56_28_10.pkl.gz",
                                            "inputs_targets_train_14x14.pkl.gz",
                                            "inputs_targets_valid_14x14.pkl.gz",
                                            "lnn_14x14_196_56_28_10.pkl.gz",
                                            "statistics_14x14_196_56_28_10.txt",
                                            iterations)

def iterate_function(f, iterations=1):
    for i in xrange(0, iterations):
        print("Nr. "+str(i))
        f(5)
