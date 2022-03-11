#! /usr/bin/python2.7

from NeuralNetworkDecimalMultiprocess import NeuralNetwork
from TrainedNetwork import TrainedNetwork
from DigitRecognition import save_inputs_targets, \
                             save_new_network, \
                             load_pkl_file, \
                             main_digits_learning, \
                             main_digits_testing, \
                             main_digits_learning_testing_iterations

import os

picture_sizes = [str(i)+"x"+str(i) for i in [28, 14, 7]]
file_extension = ".pkl.gz"
network_names = ["lnn_"+ps+file_extension for ps in picture_sizes]
inputs_targets_train_names = ["inputs_targets_train_"+ps+file_extension for ps in picture_sizes]
inputs_targets_valid_names = ["inputs_targets_valid_"+ps+file_extension for ps in picture_sizes]

def learn_test_network_1(iterations=1):
    for net_name, inp_targ_train_name, inp_targ_valid_name, i in \
        zip(network_names, inputs_targets_train_names, inputs_targets_valid_names, xrange(0, len(picture_sizes))):
        if not os.path.isfile(inp_targ_train_name):
            print("creating new inputs-targets-file for train")
            inputs_targets = save_inputs_targets(inp_targ_train_name, 50000, 2**i, "train")
        elif not os.path.isfile(net_name):
            inputs_targets = load_pkl_file(inp_targ_train_name)
        # if

        if not os.path.isfile(net_name):
            save_new_network(net_name, inputs_targets, 2**i)
        else:
            print("not creating new network!")
        # if

        if not os.path.isfile(inp_targ_valid_name):
            print("creating new inputs-targets-file for valid")
            save_inputs_targets(inp_targ_valid_name, 10000, 2**i, "valid")
        # if
    # for

    # print("learning network[1] with size 14x14")
    # main_digits_learning(network_names[1],
    #                      # inputs_targets_valid_names[1],
    #                      inputs_targets_train_names[1],
    #                      network_names[1]) #"lnn_"+picture_sizes[1]+"_iter_"+str(1)+file_extension)

    # print("testing the learned network[1]")
    # main_digits_testing(network_names[1],
    #                     inputs_targets_train_names[1], # inputs_targets_valid_names[1],
    #                     "lnn_"+picture_sizes[1]+"_statistics.txt")

    # main_digits_learning_testing_iterations(network_path_input, inputs_targets_1_path, inputs_targets_2_path, network_path_output, output_statistics_path, iterations=1)
    main_digits_learning_testing_iterations(network_names[1],
                                            inputs_targets_train_names[1],
                                            inputs_targets_train_names[1],
                                            network_names[1],
                                            "lnn_"+picture_sizes[1]+"_statistics.txt",
                                            iterations)
# def learn_network_1

def learn_test_network_1_loop(num=1):
    for _ in xrange(0, num):
        learn_test_network_1(num)
    # for
# def learn_network_1_loop

def learn_network_1():
    nn = main_digits_learning(network_names[1],
                         inputs_targets_valid_names[1],
                         network_names[1]) #"lnn_"+picture_sizes[1]+"_iter_"+str(1)+file_extension)
    return nn
# def learn_network_1


# network = load_pkl_file(network_names[1])
# inputs_targets = load_pkl_file(inputs_targets_valid_names[1])

# inputs  = inputs_targets[0]
# targets = inputs_targets[1]
# values  = inputs_targets[2]


# tn1 = TrainedNetwork()
# # nn1 = NeuralNetwork()

# tn1.set_file_path("lnn1.gz")
# # nn1.test_function()
# # tn1.set_network(nn1)

# tn2 = TrainedNetwork()
# # nn2 = NeuralNetwork()

# tn2.set_file_path("lnn2.gz")
# # nn2.test_function()
# # tn2.set_network(nn2)

# # tn1.save_network()
# # tn2.save_network()

# tn1.load_network()
# tn2.load_network()
