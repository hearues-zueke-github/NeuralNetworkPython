#! /usr/bin/python2.7

from __init__ import *
import matplotlib
matplotlib.rcParams['backend'] = 'TkAgg'
import matplotlib.pyplot as plt
import numpy as np
import Utils as utils
from NeuralNetworkDecimalMultiprocess import NeuralNetwork

nl = [4, 30, 2]
amount = 5000
inputs, targets = utils.get_n_to_2_logistic_samples(nl[0], amount)

inputs_train, inputs_test = np.split(inputs, 2)
targets_train, targets_test = np.split(targets, 2)

nn = NeuralNetwork()
nn.set_neuron_list(nl)
nn.init_random_weights()
bs, ws = nn.get_biases(), nn.get_weights()
outputs = nn.calculate_forward_many(inputs, bs, ws)

# Now try some iterations for improving the network!!!
iterations = 1000

## yyyy_mm_dd_HH_MM_SS
datetime_now = (datetime.datetime.utcnow()+datetime.timedelta(hours=2)).strftime("%Y_%m_%d_%H_%M_%S_%f")[:-3]
directory = "figures/"+datetime_now+"/"
utils.check_create_dir(directory)

def do_the_plots(directory, add_name, bs, ws, inputs_train, targets_train, inputs_test, targets_test, errors_train, errors_test):
    """  """
    """ General plots """
    plt.figure()
    plt.title("Etha-Per-Iteration")
    plt.plot(etha_per_iteration[1], etha_per_iteration[0], ".b")
    ax = plt.gca()
    ax.set_yscale("log", nonposy='clip')
    plt.savefig(directory+add_name+"etha_per_iterations.png", format="png", dpi=500)

    plt.figure()
    plt.title("Error plot curve")
    plt.plot(np.arange(len(errors_train)), errors_train, "-b")
    plt.plot(np.arange(len(errors_test)), errors_test, "-g")
    plt.xlabel("Iterations")
    plt.ylabel("Error (sqrt-mean-squared-error)")
    plt.savefig(directory+add_name+"sqrt_mean_squared_error.png", format="png", dpi=500)

    """ For Train set """
    outputs = nn.calculate_forward_many(inputs_train, bs, ws)
    mse_train_separate = nn.mean_square_error_separate(outputs, targets_train)
    mse_train_separate_x = np.arange(mse_train_separate.shape[0])
    width = 1./10/2
    bins_train = np.arange(0, 1.+width, width)
    bins_plot_x_train = bins_train[:-1]+width*.5
    histogram_train = np.histogram(mse_train_separate, bins=bins_train)[0]
    sorted_mse_train_x = np.arange(mse_train_separate.shape[0])
    sorted_mse_train_y = np.sort(mse_train_separate)
    mse_train_first_derivative_x = sorted_mse_train_x[:-1]+0.5
    mse_train_first_derivative_y = np.array([x2-x1 for x1, x2 in zip(sorted_mse_train_y[:-1], sorted_mse_train_y[1:])])
    #
    # plt.figure()
    # plt.title("Train samples mean-square-error")
    # plt.plot(mse_train_separate_x, mse_train_separate, ".b")
    # plt.savefig(directory+add_name+"train_samples_mse.png", format="png", dpi=500)
    #
    # plt.figure()
    # plt.title("Train samples mean-square-error sorted")
    # plt.plot(sorted_mse_train_x, sorted_mse_train_y, ".b", markersize=5)
    # plt.savefig(directory+add_name+"train_samples_mse_sorted.png", format="png", dpi=500)
    #
    # plt.figure()
    # plt.title("Train samples mean-square-error first derivation")
    # ax = plt.gca()
    # ax.set_yscale("log", nonposy='clip')
    # plt.plot(mse_train_first_derivative_x, mse_train_first_derivative_y, "^b", markersize=5)
    # plt.savefig(directory+add_name+"train_samples_mse_first_derivation.png", format="png", dpi=500)
    #
    # plt.figure()
    # plt.title("Train samples histogram")
    # ax = plt.gca()
    # ax.set_xticks(bins_train)
    # ax.set_xticks(bins_plot_x_train, minor=True)
    # ax.set_xlim([0, 1.])
    # ax.bar(bins_train[:-1], histogram_train, width, color='b') #, yerr=stdderiv)
    # plt.savefig(directory+add_name+"train_samples_histogram.png", format="png", dpi=500)

    """ Merge all plots in one for TRAIN set """
    f, axarr = plt.subplots(2, 2, figsize=(18,12))

    ax = axarr[0, 0]
    ax.set_title("Train samples mean-square-error")
    ax.set_xlabel("Each train feature sample vector")
    ax.set_ylabel("Mean-Square-Error of each feature sample")
    ax.plot(mse_train_separate_x, mse_train_separate, ".b")

    ax = axarr[0, 1]
    ax.set_title("Train samples mean-square-error sorted")
    ax.set_xlabel("Sorted Mean-Square-Error by the mse value")
    ax.plot(sorted_mse_train_x, sorted_mse_train_y, ".b", markersize=5)

    ax = axarr[1, 0]
    ax.set_title("Train samples mean-square-error first derivation")
    ax.set_yscale("log", nonposy='clip')
    ax.set_xlabel("")
    ax.plot(mse_train_first_derivative_x, mse_train_first_derivative_y, "^b", markersize=5)

    ax = axarr[1, 1]
    ax.set_title("Train samples histogram")
    ax.set_xticks(bins_train)#, rotation="vertical")
    ax.set_xticks(bins_plot_x_train, minor=True)
    ax.set_xlim([0, 1.])
    ax.bar(bins_train[:-1], histogram_train, width, color='b') #, yerr=stdderiv)

    plt.savefig(directory+add_name+"merged_plots_train.png", format="png", dpi=500)

    """ For Test set """
    outputs = nn.calculate_forward_many(inputs_test, bs, ws)
    mse_test_separate = nn.mean_square_error_separate(outputs, targets_test)
    mse_test_separate_x = np.arange(mse_train_separate.shape[0])
    width = 1./10/2
    bins_test = np.arange(0, 1.+width, width)
    bins_plot_x_test = bins_test[:-1]+width*.5
    histogram_test = np.histogram(mse_test_separate, bins=bins_train)[0]
    sorted_mse_test_x = np.arange(mse_test_separate.shape[0])
    sorted_mse_test_y = np.sort(mse_test_separate)
    mse_test_first_derivative_x = sorted_mse_test_x[:-1]+0.5
    mse_test_first_derivative_y = np.array([x2-x1 for x1, x2 in zip(sorted_mse_test_y[:-1], sorted_mse_test_y[1:])])
    #
    # plt.figure()
    # plt.title("Test samples mean-square-error")
    # plt.plot(np.arange(mse_test_separate.shape[0]), mse_test_separate, ".g")
    # plt.savefig(directory+add_name+"test_samples_mse.png", format="png", dpi=500)
    #
    # plt.figure()
    # plt.title("Test samples mean-square-error sorted")
    # plt.plot(sorted_mse_test_x, sorted_mse_test_y, ".g", markersize=5)
    # plt.savefig(directory+add_name+"test_samples_mse_sorted.png", format="png", dpi=500)
    #
    # plt.figure()
    # plt.title("Test samples mean-square-error first derivation")
    # ax = plt.gca()
    # ax.set_yscale("log", nonposy='clip')
    # plt.plot(mse_test_first_derivative_x, mse_test_first_derivative_y, "^g", markersize=5)
    # plt.savefig(directory+add_name+"test_samples_mse_first_derivation.png", format="png", dpi=500)
    #
    # plt.figure()
    # plt.title("Test samples histogram")
    # ax = plt.gca()
    # ax.set_xticks(bins_test)
    # ax.set_xticks(bins_plot_x_test, minor=True)
    # ax.set_xlim([0, 1.])
    # ax.bar(bins_test[:-1], histogram_test, width, color='g') #, yerr=stdderiv)
    # plt.savefig(directory+add_name+"test_samples_histogram.png", format="png", dpi=500)

    """ Merge all plots in one for TEST set """
    f, axarr = plt.subplots(2, 2, figsize=(18,12))

    ax = axarr[0, 0]
    ax.set_title("Test samples mean-square-error")
    ax.set_xlabel("Each test feature sample vector")
    ax.set_ylabel("Mean-Square-Error of each feature sample")
    ax.plot(mse_test_separate_x, mse_test_separate, ".g")

    ax = axarr[0, 1]
    ax.set_title("Test samples mean-square-error sorted")
    ax.set_xlabel("Sorted Mean-Square-Error by the mse value")
    ax.plot(sorted_mse_test_x, sorted_mse_test_y, ".g", markersize=5)

    ax = axarr[1, 0]
    ax.set_title("Test samples mean-square-error first derivation")
    ax.set_yscale("log", nonposy='clip')
    ax.set_xlabel("")
    ax.plot(mse_test_first_derivative_x, mse_test_first_derivative_y, "^g", markersize=5)

    ax = axarr[1, 1]
    ax.set_title("Train samples histogram")
    ax.set_xticks(bins_test)#, rotation="vertical")
    ax.set_xticks(bins_plot_x_test, minor=True)
    ax.set_xlim([0, 1.])
    ax.bar(bins_test[:-1], histogram_test, width, color='g') #, yerr=stdderiv)

    plt.savefig(directory+add_name+"merged_plots_test.png", format="png", dpi=500)

# components, errors_train, errors_test, etha_per_iteration = nn.improve_network_better([nl, bs, ws], inputs_train, targets_train,
#                                                                                       inputs_test, targets_test, iterations, 0.01, copy=True, withrandom=False)
# nl, bs, ws = components
# do_the_plots(directory, "without_random", bs, ws, inputs_train, targets_train, inputs_test, targets_test, errors_train, errors_test)

# TODO make a function for creating a random network
# TODO make a function for iterating the network, after iterations save all data and update the diagrams

components, errors_train, errors_test, etha_per_iteration =\
    nn.improve_network_better([nl, bs, ws], inputs_train, targets_train,
                              inputs_test, targets_test, iterations, 0.0001,
                              copy=True, withrandom=True)
nl, bs, ws = components
do_the_plots(directory, "", bs, ws, inputs_train, targets_train, inputs_test, targets_test, errors_train, errors_test)
