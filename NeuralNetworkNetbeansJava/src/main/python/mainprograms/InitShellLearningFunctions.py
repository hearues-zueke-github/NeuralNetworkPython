#! /usr/bin/python2.7

import os
# os.sched_setaffinity(0, {0})

import CreateGraphs as cgraphs
import Utils as utils

from NeuralNetworkDecimalMultiprocess import NeuralNetwork
from TrainedNetwork import TrainedNetwork
from DigitRecognition import main_digits_learning_testing_iterations, \
                             main_binary_adder_learning_testing_iterations

import os

def learn_14x14_neural_network(neuron_list, iterations=1, is_autoencoder=False):
    name_part = "".join(map(lambda x: "_"+str(x), neuron_list))

    path_neural_network_in  = "lnn_14x14"+("_autoencoder" if is_autoencoder else "")+name_part+".pkl.gz"
    path_train = "inputs_targets_14x14"+("_autoencoder" if is_autoencoder else "")+"_train.pkl.gz"
    path_valid = "inputs_targets_14x14"+("_autoencoder" if is_autoencoder else "")+"_valid.pkl.gz"
    path_neural_network_out = path_neural_network_in
    path_statistics = "statistics_14x14"+("_autoencoder" if is_autoencoder else "")+name_part+".txt"

    main_digits_learning_testing_iterations(path_neural_network_in,
                                            path_train,
                                            path_valid,
                                            path_neural_network_out,
                                            path_statistics,(neuron_list, bits, learning_rate, iterations=1, is_autoencoder=False):
    name_part = "".join(map(lambda x: "_"+str(x), neuron_list))

    path_neural_network = "lnn_binadder"+name_part+".pkl.gz"
    path_inputs_targets = "inputs_targets_binary_"+str(bits)+"_bits.pkl.gz"
    path_statistics = "statistics_b
                                            iterations,
                                            is_save_error=is_autoencoder)

def learn_binary_adder_neural_networkinary_"+str(bits)+"_bits"+name_part+".txt"

    main_binary_adder_learning_testing_iterations(path_neural_network,
                                                  path_inputs_targets,
                                                  learning_rate,
                                                  path_statistics,
                                                  iterations)

def iterate_function(tupl, iterations=1):
    "tupl = (f, *a, **k)"
    for i in xrange(0, iterations):
        print("Nr. "+str(i))
        tupl[0](*tupl[1], **tupl[2])


# All normal Networks for learning
def f1():
    func = iterate_function
    tupl = (learn_14x14_neural_network, [[14*14, 28, 10], 2], {"is_autoencoder": False})
    kwargs = {"iterations": 50}

    taken_time = utils.test_time_args_direct(func, tupl, kwargs)
    return taken_time
def f2():
    # iterate_function((learn_14x14_neural_network, [[14*14, 56, 28, 10], 5], {"is_autoencoder": False}), iterations=50)
    func = iterate_function
    tupl = (learn_14x14_neural_network, [[14*14, 56, 28, 10], 2], {"is_autoencoder": False})
    kwargs = {"iterations": 100}

    taken_time = utils.test_time_args_direct(func, tupl, kwargs)
    return taken_time
def f3():
    # iterate_function((learn_14x14_neural_network, [[14*14, 98, 10], 5], {"is_autoencoder": False}), iterations=50)
    func = iterate_function
    tupl = (learn_14x14_neural_network, [[14*14, 98, 10], 2], {"is_autoencoder": False})
    kwargs = {"iterations": 225}

    taken_time = utils.test_time_args_direct(func, tupl, kwargs)
    return taken_time


# All autoencoder networks for learning
def fa1():
    func = iterate_function
    tupl = (learn_14x14_neural_network, [[14*14, 14*2, 10, 14*2, 14*14], 2], {"is_autoencoder": True})
    kwargs = {"iterations": 200}

    taken_time = utils.test_time_args_direct(func, tupl, kwargs)
    return taken_time

def fa2():
    func = iterate_function
    tupl = (learn_14x14_neural_network, [[14*14, 14*4, 10, 14*4, 14*14], 2], {"is_autoencoder": True})
    kwargs = {"iterations": 200}

    taken_time = utils.test_time_args_direct(func, tupl, kwargs)
    return taken_time

def fa3():
    func = iterate_function
    tupl = (learn_14x14_neural_network, [[14*14, 14*4, 14*2, 10, 14*2, 14*4, 14*14], 2], {"is_autoencoder": True})
    kwargs = {"iterations": 200}

    taken_time = utils.test_time_args_direct(func, tupl, kwargs)
    return taken_time


def fbinlearn(bits, etha, iter1, iter2):
    func = iterate_function
    tupl = (learn_binary_adder_neural_network, [[bits*2, bits*3, bits+1], bits, etha, iter1], {"is_autoencoder": False})
    kwargs = {"iterations": iter2}

    taken_time = utils.test_time_args_direct(func, tupl, kwargs)
    print("Taken time for calculation: "+str(taken_time)+" s")

def fbin2():
    fbinlearn(2, 0.1, 100, 1)

def fbin3():
    fbinlearn(3, 0.005, 100, 1)

def fbin4():
    fbinlearn(4, 0.005, 100, 1)

def fbin5():
    fbinlearn(5, 0.005, 100, 1)

def fbin6():
    fbinlearn(6, 0.0005, 1, 1)

def fbin8():
    fbinlearn(8, 0.00002, 10, 5000)

def fbin2_with_output_targets(bits, learning_rate, iterations=1, offset=0):
    cgraphs.get_output_targets_binary_adder_plot_points(bits, iterations=offset, show_not_save_file=False)
    # learning_rate = 0.1
    steps = 1000
    for i in xrange(0, iterations):
        print("Learn and plot output targets Nr. "+str(i))
        fbinlearn(bits, learning_rate, steps, 1)
        cgraphs.get_output_targets_binary_adder_plot_points(bits, iterations=(i+1)*steps+offset, show_not_save_file=False)
    # for
# def fbin2_with_output_targets


# def fbin4():
#     func = iterate_function
#     tupl = (learn_binary_adder_neural_network, [[8, 12, 5], 4, 5], {"is_autoencoder": False})
#     kwargs = {"iterations": 3}

#     taken_time = utils.test_time_args_direct(func, tupl, kwargs)
#     print("Taken time for calculation: "+str(taken_time)+" s")

# def fbin5():
#     func = iterate_function
#     tupl = (learn_binary_adder_neural_network, [[10, 15, 6], 5, 0.005, 10], {"is_autoencoder": False})
#     kwargs = {"iterations": 200}

#     taken_time = utils.test_time_args_direct(func, tupl, kwargs)
#     print("Taken time for calculation: "+str(taken_time)+" s")

# def fbin6():
#     func = iterate_function
#     tupl = (learn_binary_adder_neural_network, [[12, 18, 7], 6, 0.001, 5], {"is_autoencoder": False})
#     kwargs = {"iterations": 145}

#     taken_time = utils.test_time_args_direct(func, tupl, kwargs)
#     print("Taken time for calculation: "+str(taken_time)+" s")

# def fbin8():
#     func = iterate_function
#     tupl = (learn_binary_adder_neural_network, [[16, 24, 9], 8, 0.001, 5], {"is_autoencoder": False})
#     kwargs = {"iterations": 1}

#     taken_time = utils.test_time_args_direct(func, tupl, kwargs)
#     print("Taken time for calculation: "+str(taken_time)+" s")


# One Function for all functions (not very well, because of serial procedure)!
def f_all():
    t1 = f1()
    t2 = f2()
    t3 = f3()

    print("t1 = "+str(t1))
    print("t2 = "+str(t2))
    print("t3 = "+str(t3))

def fa_all():
    t1 = fa1()
    t2 = fa2()
    t3 = fa3()

    print("t1 = "+str(t1))
    print("t2 = "+str(t2))
    print("t3 = "+str(t3))
