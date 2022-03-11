#! /usr/bin/python2.7

from __init__ import *

# import sys
# import os
# import shutil

# import numpy as np
# import Utils as utils

# import matplotlib
# from matplotlib import pyplot as plt

# matplotlib.use("Agg")

print_variable = lambda variable_name: utils.print_variable(variable_name, globals())

def get_plot_of_14x14_network(list_of_neuron_list):
    pos_lists = []
    error_lists = []

    names = ["".join(map(lambda x: "_"+str(x), neuron_list)) for neuron_list in list_of_neuron_list]
    file_names = ["statistics_14x14"+name+".txt" for name in names]
    # with open("statistics_14x14_196_28_10.txt", "rb") as fin:
    for fname in file_names:
        pos_lists.append([])
        error_lists.append([])
        with open(fname, "rb") as fin:
            for line, i in zip(fin, xrange(0, 1000)):

                elems = line.split()

                if i != 0:
                    pos_lists[-1].append(float(elems[2]))
                    error_lists[-1].append(float(elems[4][:-1]))
                # if
            # for
        # with

    # for i in xrange(0, len(error_lists)):
    #     utils.print_variable("error_lists["+str(i)+"]", locals())

    p_x = []
    p_y_pos = []
    p_y_error = []
    for pos_l, error_l in zip(pos_lists, error_lists):
        p_x.append([i for i in xrange(0, len(error_l))])
        p_y_pos.append(pos_l)
        p_y_error.append(error_l)

    # k, d = np.polyfit(p_x[24:229], p_y_error[24:229], 1)
    # utils.print_variable("k", locals())
    # utils.print_variable("d", locals())

    # p_l_x = [24, 229]
    # p_l_y_pos = [k*24+d, k*229+d]
    # p_l_y_error = [k*24+d, k*229+d]

    for i in xrange(0, len(pos_lists)):
        plot_pos = plt.figure()
        axes = plt.gca()
        bottom_y = 0
        top_y = 10000
        axes.set_ylim([bottom_y, top_y])
        if i == 0:
            axes.set_xlim([0, 800])
        elif i == 1:
            axes.set_xlim([0, 850])
        elif i == 2:
            axes.set_xlim([0, 600])

        plt.grid(True)

        plt.title("Positive learned Values\nNetwork layers: "+str(list_of_neuron_list[i])+", Learning-Rate = 0.0005")
        plt.xlabel("Iteration / Epoch")
        plt.ylabel("Learned Digits of Total 10.000 images")
        plt.plot(p_x[i], p_y_pos[i])
        plt.savefig("pictures/graph"+names[i]+"_pos.eps", format="eps")
        # plt.savefig("pictures/graph"+names[i]+"_pos.png", format="png")

        plot_error = plt.figure()
        axes = plt.gca()
        top_y = 2.
        axes.set_ylim([0., top_y])
        if i == 0:
            axes.set_xlim([0, 800])
        elif i == 1:
            axes.set_xlim([0, 850])
        elif i == 2:
            axes.set_xlim([0, 600])

        plt.grid(True)

        plt.title("Error learning curve\nNetwork layers: "+str(list_of_neuron_list[i])+", Learning-Rate = 0.0005")
        plt.xlabel("Iteration / Epoch")
        plt.ylabel("Total Error in %")
        # plt.yscale("log", base=2)
        plt.plot(p_x[i], p_y_error[i])
        plt.savefig("pictures/graph"+names[i]+"_error.eps", format="eps")
        # plt.savefig("pictures/graph"+names[i]+"_error.png", format="png")
    # plt.plot(p_l_x, p_l_y_error, marker="o", color="r", ls="dotted")

    # plt.yscale("log")
    # plt.show(block = False)

    # print("Press ENTER for finish...")
    # a = raw_input()

def get_plot_of_14x14_autoencoder(list_of_neuron_list):
    # pos_lists = []
    error_lists = []

    names = ["".join(map(lambda x: "_"+str(x), neuron_list)) for neuron_list in list_of_neuron_list]
    file_names = ["statistics_14x14_autoencoder"+name+".txt" for name in names]

    for fname in file_names:
        error_lists.append([])
        with open(fname, "rb") as fin:
            for line, i in zip(fin, xrange(0, 1600)):

                elems = line.split()

                if i != 0:
                    error_lists[-1].append(float(elems[2][:-1]))
                # if
            # for
        # with
    # for

    p_x = []
    p_y_error = []
    for error_l in error_lists:
        p_x.append([i for i in xrange(0, len(error_l))])
        p_y_error.append(error_l)
    # for

    for i in xrange(0, len(error_lists)):

        plot_error = plt.figure()
        axes = plt.gca()
        # _, top_y = axes.get_ylim()
        top_y = 50
        axes.set_ylim([0., top_y])
        axes.set_aspect(10)

        plt.grid(True)

        plt.title("Error learning curve of autoencoder\nNetwork layers: "+str(list_of_neuron_list[i])+", Learning-Rate = 0.00005")
        plt.xlabel("Iteration / Epoch")
        plt.ylabel("Total Error in %")
        plt.plot(p_x[i], p_y_error[i])
        # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        plt.savefig("pictures/graph_autoencoder"+names[i]+"_error.eps", format="eps",bbox_inches='tight')
        # plt.savefig("pictures/graph_autoencoder"+names[i]+"_error.png", format="png")
    # for
# def get_plot_of_14x14_autoencoder(

def get_plot_of_binary_adder_network(list_of_neuron_list, list_of_bits):
    pos_lists = []
    error_lists = []

    names = ["".join(map(lambda x: "_"+str(x), neuron_list)) for neuron_list in list_of_neuron_list]
    file_names = ["statistics_binary_"+str(bits)+"_bits"+name+".txt" for bits, name in zip(list_of_bits, names)]
    for fname in file_names:
        pos_lists.append([])
        error_lists.append([])
        with open(fname, "rb") as fin:
            for line, i in zip(fin, xrange(0, 1000000)):

                elems = line.split()

                if i != 0:
                    pos_lists[-1].append(float(elems[2]))
                    error_lists[-1].append(float(elems[4][:-1]))
                # if
            # for
        # with

    p_x = []
    p_y_pos = []
    p_y_error = []
    for pos_l, error_l in zip(pos_lists, error_lists):
        p_x.append([i for i in xrange(0, len(error_l))])
        p_y_pos.append(pos_l)
        p_y_error.append(error_l)


    nn_names = ["lnn_binadder"+name+".pkl.gz" for name in names]
    list_of_learning_rate = [utils.load_pkl_file(name).get_learning_rate() for name in nn_names]

    for i, bits, learning_rate in zip(xrange(0, len(pos_lists)), list_of_bits, list_of_learning_rate):
        plot_pos = plt.figure()
        axes = plt.gca()
        top_y = 2**(2*bits)+1
        axes.set_ylim([0., top_y])
        top_x = len(p_x[i])
        axes.set_xlim(0, top_x)

        plt.grid(True)

        plt.title("Positive learned Values for a "+str(bits)+" bit Adder\nNetwork layers: "+str(list_of_neuron_list[i])+", Learning-Rate = "+str(learning_rate))
        plt.xlabel("Iteration / Epoch")
        plt.ylabel("Learned "+str(2**(bits*2))+" Additions")
        plt.plot(p_x[i], p_y_pos[i])
        plt.savefig("pictures/graph_"+str(bits)+"_bits_binary_adder"+names[i]+"_pos.eps", format="eps")
        # plt.savefig("pictures/graph_"+str(bits)+"_bits_binary_adder"+names[i]+"_pos.png", format="png")

        plot_error = plt.figure()
        axes = plt.gca()
        top_y = 4.
        axes.set_ylim([0., top_y])
        top_x = len(p_x[i])
        axes.set_xlim(0, top_x)

        plt.grid(True)

        plt.title("Error learning curve for a "+str(bits)+" bit Adder\nNetwork layers: "+str(list_of_neuron_list[i])+", Learning-Rate = "+str(learning_rate))
        plt.xlabel("Iteration / Epoch")
        plt.ylabel("Total Error in %")
        # plt.yscale("log", base=2)
        plt.plot(p_x[i], p_y_error[i])
        plt.savefig("pictures/graph_"+str(bits)+"_bits_binary_adder"+names[i]+"_error.eps", format="eps")
        # plt.savefig("pictures/graph_"+str(bits)+"_bits_binary_adder"+names[i]+"_error.png", format="png")
    # plt.plot(p_l_x, p_l_y_error, marker="o", color="r", ls="dotted")

    # plt.yscale("log")
    # plt.show(block = False)

    # print("Press ENTER for finish...")
    # a = raw_input()

list_of_neuron_list = [[196, 56, 28, 10],
                           [196, 28, 10],
                           [196, 98, 10]]
list_of_neuron_list_autoencoder = [[196, 28, 10, 28, 196],
                                   [196, 56, 10, 56, 196],
                                   [196, 56, 28, 10, 28, 56, 196]]

# get_plot_of_14x14_network(list_of_neuron_list)
# get_plot_of_14x14_autoencoder(list_of_neuron_list_autoencoder)
get_plot_of_binary_adder_network([#[4, 6, 3],
                                  #[6, 9, 4],
                                  #[8, 12, 5],
                                  #[10, 15, 6],
                                  #[12, 18, 7],
                                  [16, 24, 9]],
                                  [8]) #[2, 3, 4, 5, 6, 8])
print("Finnish creating graphs!")

src = "./pictures"
dest = "../documents/graphs"

src_files = os.listdir(src)
for file_name in src_files:
    full_file_name = os.path.join(src, file_name)
    # print("full_file_name = "+str(full_file_name))
    if (os.path.isfile(full_file_name)):
        shutil.copy(full_file_name, dest)
print("Finish copying graphs!")
