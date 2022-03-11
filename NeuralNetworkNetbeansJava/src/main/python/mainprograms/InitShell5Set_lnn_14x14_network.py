#! /usr/bin/python2.7

import sys
import os

import Utils as utils

from TrainedNetwork import TrainedNetwork
from NeuralNetworkDecimalMultiprocess import NeuralNetwork

print_variable = lambda variable_name: utils.print_variable(variable_name, globals())

def save_new_random_network(neuron_list, file_path):
    nn = NeuralNetwork()
    nn.set_neuron_list(neuron_list)
    nn.init_random_weights()

    tn = TrainedNetwork()
    tn.set_file_path(file_path)
    tn.set_network(nn)
    tn.save_network()
# def save_new_random_network

def create_new_random_network(neuron_list, is_autoencoder=False, additional_names=""):
    file_path = "".join(map(lambda x: "_"+str(x), neuron_list))
    
    file_path = "lnn_14x14"+("_autoencoder" if is_autoencoder else "")+file_path+".pkl.gz"

    print_variable("neuron_list")
    print_variable("file_path")

    save_new_random_network(neuron_list, file_path)
# def create_new_random_network

# neuron_list = [14*14, 14*4, 14*2, 10]
