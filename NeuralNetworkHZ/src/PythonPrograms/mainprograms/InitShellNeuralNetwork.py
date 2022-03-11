#! /usr/bin/python2.7

# from NeuralNetworkMultiprocess import NeuralNetwork
from NeuralNetworkDecimalMultiprocess import NeuralNetwork
from TrainedNetwork import TrainedNetwork

from decimal import Decimal as dec
from copy import deepcopy

import numpy as np
import BinaryAdder as binary

nn = NeuralNetwork(5, 3, 20)

nn.set_neuron_list([5,4,3])
nn.init_random_weights()
nn.init_random_targets()

nl = nn.neuron_list

bs = nn.biases
ws = nn.weights

inputs = nn.inputs
targets = nn.targets

bsdl, wsdl = nn.backprop_many(inputs, bs, ws, targets, dec("0.005"))
print("backprop inputs and targets.")
network = [nl, bs, ws]

network_better = nn.get_improved_network(network, bsdl, wsdl)
print("improved network")

# network_better_sgd, error_list_sgd = nn.improve_network_sgd(network, inputs, targets, 20, dec("0.005"))
# print("improved network with sgd in 10 iterations")

# network_better_bgd, error_list_bgd = nn.improve_network_bgd(network, inputs, targets, 20, dec("0.005"))
# print("improved network with bgd in 10 iterations")

# inputs_2binadder = [[0,0,0,0]]
# digits = 2
# inp_orig, targ_orig = get_inputs_targets_2binadder(digits)
# d = {0: 0.1, 1: 0.9}
# inp = [get_changed_list_vals_dict(l, d) for l in inp_orig]
# targ = [get_changed_list_vals_dict(l, d) for l in targ_orig]
# inp = [np.transpose(np.array([i])) for i in inp]
# targ = [np.transpose(np.array([t])) for t in targ]

digits = 4

inp_orig, targ_orig = binary.get_inputs_targets_2binadder(digits)
inputs, targets = binary.get_binaryadder_inputs_targets(digits)

nn.neuron_list = [2*digits, 4*digits, digits+1]

nn.init_random_weights()
nn.inputs = inputs
nn.targets = targets

tn = TrainedNetwork()
tn.set_file_path("my_network.nn")
tn.load_network()
