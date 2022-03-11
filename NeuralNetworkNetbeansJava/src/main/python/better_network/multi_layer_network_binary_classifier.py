#! /usr/bin/python2.7
# -*- coding: utf-8 -*-

from apt import utils

import colorama
import csv
import dill
import gzip
import os
import socket
import sys

import cProfile

import cPickle as pkl
import numpy as np

from copy import deepcopy
from PIL import Image

sys.path.insert(0, os.path.abspath('../utils'))
import HashFunction as hfunc
import MathUtils
import Utils
import UtilsBinary

import binary_matrix_data_set

from neuralnetwork import NeuralNetwork
"""
    TODO:
    20.06.2017: Make a distunction between BGD, SGD, mini BGD, and multi mini BGD (16, 32, 64, etc.)
"""

def train_mnist_classifier_better(data_size, nl_hidden, with_momentum_1_degree, str_hidden_function):
    file_path_data_set = "mnist_{}x{}_tr_noise.npz.gz".format(data_size, data_size)
    
    hostname = socket.gethostname()
    if "figi" in hostname:
        home = "/calc/students/ziko/nn_learning/neuronal_network"
        full_path = home+"/networks/mnist_networks"
        bws_folder = home+"/bws_arrays"
    else:
        home = os.path.expanduser("~")
        full_path = home+"/Documents/saved_networks/mnist_networks"
        bws_folder = home+"/Documents/bws_arrays"

    assert len(nl_hidden) == 1
    nl = [data_size**2]+nl_hidden+[10]
    nl_str = "_".join(list(map(str, nl)))

    network_path = full_path+"/data_size_{}x{}_nl_{}_with_momentum_1_degree_{}_func_{}".format(
        data_size, data_size, nl_str, with_momentum_1_degree, str_hidden_function)

    if not os.path.exists(network_path):
        os.makedirs(network_path)

    file_path_network = network_path+"/whole_network.pkl.gz"
    if not os.path.isfile(file_path_network):
        nn = NeuralNetwork()
        
        nn.bws = Utils.create_new_bws(bws_folder, nl)
        nn.nl = nl
        nn.nl_str = nl_str
        nn.calc_cost = nn.f_cecf
        nn.calc_missclass = nn.f_missclass_onehot
        nn.calc_missclass_vector = nn.f_missclass_onehot_vector
        nn.with_momentum_1_degree = with_momentum_1_degree
        nn.with_confusion_matrix = True
        nn.set_hidden_function(str_hidden_function)

        with gzip.GzipFile(file_path_network, "wb") as f:
            dill.dump(nn, f)

    NeuralNetwork.do_step_by_step(network_path, 20, 10, file_path_data_set, bws_folder)

def train_number_matrix_multiply_regression(inp_amount, out_amount, nl_hidden, with_momentum_1_degree, str_hidden_function):
    file_path_data_set = binary_matrix_data_set.get_file_path_data_set(8000, inp_amount, out_amount)

    hostname = socket.gethostname()
    if "figi" in hostname:
        home = "/calc/students/ziko/nn_learning/neuronal_network"
        full_path_network = home+"/matrix_multiply_networks"
        bws_folder = home+"/bws_arrays"
    else:
        home = os.path.expanduser("~")
        full_path_network = home+"/Documents/saved_networks/matrix_multiply_networks"
        bws_folder = home+"/Documents/bws_arrays"

    assert len(nl_hidden) == 1
    nl = [inp_amount]+nl_hidden+[out_amount]
    nl_str = "_".join(list(map(str, nl)))
    network_path = full_path_network+"/nl_{}_with_momentum_1_degree_{}_func_{}".format(
        nl_str, str(with_momentum_1_degree).lower(), str_hidden_function)

    full_path_learned_weights = network_path+"/learned_weights"

    if not os.path.exists(network_path):
        os.makedirs(network_path)

    file_path_network = network_path+"/whole_network.pkl.gz"
    if not os.path.isfile(file_path_network):
        nn = NeuralNetwork()

        nn.bws = Utils.create_new_bws(bws_folder, nl)
        nn.nl = nl
        nn.nl_str = nl_str
        nn.calc_cost = nn.f_cecf
        nn.calc_missclass = nn.f_missclass
        nn.calc_missclass_vector = nn.f_missclass_vector
        nn.with_momentum_1_degree = with_momentum_1_degree
        nn.with_confusion_matrix = False
        nn.set_hidden_function(str_hidden_function)
        
        with gzip.GzipFile(file_path_network, "wb") as f:
            dill.dump(nn, f)

    NeuralNetwork.do_step_by_step(network_path, 30, 20, file_path_data_set, bws_folder)

if __name__ == "__main__":
    if not os.path.exists("pictures"):
        os.makedirs("pictures")

    argv = sys.argv
    if len(argv) < 5:
        print("less than 5 arguments!")
        print("First args: mnist, multmatrix")
        sys.exit(-1)

    used_function = argv[1]
    print("used_function: {}".format(used_function))

    if used_function != "mnist" and \
       used_function != "multmatrix":
        print("wrong used_function!")
        sys.exit(-1)

    argv = argv[2:]

    if used_function == "mnist":
        if len(argv) < 4:
            print("no datasize!")
            sys.exit(-1)
        data_size = int(argv[0])
        argv = argv[1:]
    elif used_function == "multmatrix":
        if len(argv) < 4:
            print("no inp,out amount given!")
            sys.exit(-1)
        inp_amount, out_amount = list(map(int, argv[0].split(",")))
        argv = argv[1:]

    str_nl_hidden = argv[0]
    str_with_momentum_1_degree = argv[1]
    str_hidden_function = argv[2]

    hidden_functions = ["sig", "tanh", "relu"]
    if not str_hidden_function in hidden_functions:
        print("no correct defined function given!")
        sys.exit(-1)

    print("str_nl_hidden: {}".format(str_nl_hidden))
    print("str_with_momentum_1_degree: {}".format(str_with_momentum_1_degree))
    print("str_hidden_function: {}".format(str_hidden_function))

    nl_hidden = list(map(int, str_nl_hidden.split(",")))
    with_momentum_1_degree = bool(int(str_with_momentum_1_degree))

    if used_function == "mnist":
        train_mnist_classifier_better(data_size, nl_hidden, with_momentum_1_degree, str_hidden_function)
    elif used_function == "multmatrix":
        train_number_matrix_multiply_regression(inp_amount, out_amount, nl_hidden, with_momentum_1_degree, str_hidden_function)
