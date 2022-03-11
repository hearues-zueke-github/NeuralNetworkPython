#! /usr/bin/python2.7

# 2015.07.13
# ipython
# klassen erstellen
# stochastic gratienten descent (abstieg)
# batch GD
# use python 2.7!

# 2015.10.12
# - target und output vergleichen
# - biases und weights ins network speichern
# - inputs und targets entsprechend mit bgd, sgd aufrufen
# - sgd richtig machen!- 
# - graph mit targets
# - nicht notwendig eigene klassen! (auszer ich will es persoenlich so machen)
# - MNIST Datensatz \
#                    }- probieren, epochen werden lange brauchen!
# - Auto Encoder    /

import sys
from sys import stdout
stdoutw = stdout.write

import os

import numpy as np
import Tkinter as tk
import matplotlib.pyplot as plt

from random import randint
from random import uniform

from copy import deepcopy
from operator import itemgetter

import decimal
from decimal import Decimal as dec

# import collections
# compare = lambda x, y: collections.Counter(x) == collections.Counter(y)

# For colorful output in Terminal
import colorama
from colorama import Fore, Back, Style

class Test(object):

    def __init__(self, attr1):
        self.attr1 = attr1
    # def __init__
    def __str__(self):
        return str(self.__dict__)
    # def __str__
    def __eq__(self, other): 
        return self.__dict__ == other.__dict__
    # def __eq__
# class Test

# For clearing the Terminal
def cls():
    os.system('cls' if os.name == 'nt' else 'clear')

# Helpful for getting the max depth in a list
depth = lambda L: isinstance(L, list) and max(map(depth, L))+1

decimal.getcontext().prec = 10
np.set_printoptions(precision=10)

__author__ = "haris"
__date__ = "$Jun 24, 2015 11:10:57 AM$"

## Class for Network
## Class for Weights
## Class for Biases

class NeuralNetwork(Exception):

    error_not_init = "Please, initial first the matrices!\nUse e.g. init_random() or other functions"

    error_network_not_better = "This network is not better than the saved one!"
    error_network_not_saved = "Cannot load network! No Network saved!"

    def __init__(self, inputs = 1, outputs = 1, targets = 5, is_decimal = False):
        self.inputs_amount = inputs
        self.outputs_amount = outputs
        self.hidden = []
        self.matrix_init = False
        self.weights_init = False
        self.targets_init = False
        self.neuron_list = [inputs, outputs]
        self.weights = []
        self.biases = []
        
        # TODO: add a functionality for plotting and add the error plots in an other figure,!
        # To prevent collision with the output-target plot
        self.error_list = []
        self.last_error = 1.0
        self.is_prev_error_there = False
        self.prev_error = 0

        self.targets_amount = targets
        if (targets > 0):
            self.targets_amount_set = True
        else:
            self.targets_amount_set = False
        # if

        self.io_targets = []

        self.is_decimal = is_decimal
        str_num = "0.1"
        if is_decimal:
            self.learning_rate = dec(str_num)
        else:
            self.learning_rate = float(str_num)
        # if

        self.is_show_plot = True

        # for saving temporarly the last best network config! (with inputs, and targets!)
        self.is_best_network_saved = False
        self.last_best_network_default = {"neuron_list": [],
                                          "biases": [],
                                          "weights": [],
                                          "inputs": [],
                                          "targets": [],
                                          "learning_rate": 0.1,
                                          "lowest_error": 1.0}
        self.last_best_network = deepcopy(self.last_best_network_default)
    # def __init__

    def init_complete_network(self, neuronal_list, inputs, targets):
        self.neuronal_list = deepcopy(neuronal_list)
        self.inputs = deepcopy(inputs)
        self.targets = deepcopy(targets)

        self.biases = self.get_random_biases_list(self.neuronal_list)
        self.weights = self.get_random_weights_list(self.neuronal_list)
    # def init_complete_network

    def scalar_matrix_multiplication(self, scalar, matrix):
        scalar_matrix = [[float(scalar) for i2 in range(0, len(list(matrix[0])))] for i1 in range(0, len(list(matrix)))]
        scalar_matrix = np.array(scalar_matrix)
        return scalar_matrix * matrix
    # def scala_matrix_multiplication
    
    def scalar_matrix_multiplication_decimal(self, scalar, matrix):
        scalar_matrix = [[dec(scalar) for i2 in range(0, len(list(matrix[0])))] for i1 in range(0, len(list(matrix)))]
        scalar_matrix = np.array(scalar_matrix)
        return scalar_matrix * matrix
    # def scala_matrix_multiplication_decimal

    def f(self, a) : return (1 / (1 + np.exp(-a)))
    def fi(self, a) : return (self.f(a) * self.f(1 - a))
    
    def fd(self, a) : return (dec("1") / (dec("1") + dec("2.718281828459045")**(-a)))
    def fdi(self, a) : return (self.f(a) * self.f(dec("1") - a))

    def sigmoid(self, vector):
        new_vector_list = [val.tolist()[0] for val in vector]
        return np.array([[self.f(v)] for v in new_vector_list])
    # def sigmoid

    def sigmoid_invert(self, vector):
        new_vector_list = [val.tolist()[0] for val in vector]
        return np.array([[self.fi(v)] for v in new_vector_list])
    # def sigmoid_invert

    def sigmoid_decimal(self, vector):
        new_vector_list = [val.tolist()[0] for val in vector]
        return np.array([[self.fd(v)] for v in new_vector_list])
    # def sigmoid

    def sigmoid_invert_decimal(self, vector):
        new_vector_list = [val.tolist()[0] for val in vector]
        return np.array([[self.fdi(v)] for v in new_vector_list])
    # def sigmoid_invert

    def square_error(self, v1, v2):
        square_error = 0.0
        for n1, n2 in zip(v1.transpose().tolist()[0], v2.transpose().tolist()[0]):
            square_error += (float(n1) - float(n2))**2
        # for
        return square_error
    # def square_error

    def square_error_many(self, targets_calc, targets):
        square_error = 0.0
        for tc, t in zip(targets_calc, targets):
            square_error += self.square_error(tc, t)
        # for
        return square_error
    # def square_error_of_targets

    def square_error_decimal(self, v1, v2):
        square_error = dec("0.0")
        for n1, n2 in zip(v1.transpose().tolist()[0], v2.transpose().tolist()[0]):
            square_error += (dec(n1) - dec(n2))**2
        # for
        return square_error
    # def square_error

    def square_error_many_decimal(self, targets_calc, targets):
        square_error = dec("0.0")
        for tc, t in zip(targets_calc, targets):
            square_error += self.square_error_decimal(tc, t)
        # for
        return square_error
    # def square_error_of_targets

    def compare_two_networks(self, network1, network2):
        nl1, bs1, ws1 = network1[0], network1[1], network1[2]
        nl2, bs2, ws2 = network2[0], network2[1], network2[2]

        # print("nl1 = "+str(nl1))
        # print("nl2 = "+str(nl2))
        if not (Test(nl1) == Test(nl2)):
            return False
        # if

        for (b1, b2) in zip(bs1, bs2):
            for (bl1, bl2) in zip(b1, b2):
                for (bll1, bll2) in zip(bl1, bl2):
                    if not (Test(bll1) == Test(bll2)):
                        return False
                    # if
                # for
            # for
        # for

        for (w1, w2) in zip(ws1, ws2):
            for (wl1, wl2) in zip(w1, w2):
                for (wll1, wll2) in zip(wl1, wl2):
                    if not (Test(wll1) == Test(wll2)):
                        return False
                    # if
                # for
            # for
        # for

        return True
    # def compare_two_networks

    def stats(self):
        print("inputs = " + str(self.inputs_amount))
        print("outputs = " + str(self.outputs_amount))
        if len(self.hidden) == 0:
            print("No hidden layer")
        else:
            print(str(len(self.hidden)) + " layers set with " + str(self.hidden) + " neurons")
        # if
        print("Neuronlist: "+str(self.neuron_list))
        if self.matrix_init == True:
            print("Matrix is initialized:")
            print("Biases and Weights:")
            for i in xrange(0, len(self.weights)):
                print("For Layer "+str(i+1)+" with Format ("+str(self.neuron_list[i])+","+str(self.neuron_list[i+1])+")")
                print("Biase:\n"+str(self.biases[i]))
                print("Weight:\n"+str(self.weights[i]))
        else:
            print("Matrix is not initialized!!!")
        # if
 
        print("Plot is set to "+str(self.is_show_plot))
    # def stats

    # returns 0 if succeed, else -1
    def save_best_network(self):
        if self.is_best_network_saved == True:
            if self.last_best_network["lowest_error"] <= self.last_error:
                print(self.error_network_not_better)
                return -1
            # if
        else:
            self.is_best_network_saved = True
        # if

        lbn = self.last_best_network

        lbn["neuron_list"] = deepcopy(self.neuron_list)
        lbn["biases"] = deepcopy(self.biases)
        lbn["weights"] = deepcopy(self.weights)
        lbn["inputs"] = deepcopy(self.inputs)
        lbn["targets"] = deepcopy(self.targets)
        lbn["learning_rate"] = self.learning_rate
        lbn["lowest_error"] = self.last_error

        print("Network was now saved!")

        return 0
    # def

    # returns 0 if succeed, else -1
    def load_last_network(self):
        if self.is_best_network_saved == False:
            print(error_network_not_better)

            return -1
        # if

        lbn = self.last_best_network

        self.neuron_list = deepcopy(lbn["neuron_list"])
        self.biases = deepcopy(lbn["biases"])
        self.weights = deepcopy(lbn["weights"])
        self.inputs = deepcopy(lbn["inputs"])
        self.targets = deepcopy(lbn["targets"])
        self.learning_rate = lbn["learning_rate"]
        self.last_error = lbn["lowest_error"]

        print("Network was now loaded!")

        return 0
    # def

    def reset_saved_networks(self):
        self.last_best_network = deepcopy(self.last_best_network_default)
        self.is_best_network_saved = False
        print("Reseted best network!")
    # def reset_saved_networks

    def show_error_plot(self, error_list):
        p_x = [i for i in xrange(0, len(self.error_list))]
        p_y = [e for e in self.error_list]
        
        plt.figure()
        plt.plot(p_x, p_y)

        plt.yscale("log")
        plt.show(block = False)
    # def show_error_plot

    def show_own_error_plot(self):
        show_error_plot(self.error_list)
    # def show_error_plot

    def show_outputs_targets_plot_points(self, inputs, targets):
        # Get the network biases and weights
        biases = self.biases
        weights = self.weights
        # Calculate the output of this network
        outputs = self.calculate_forward_many(inputs, biases, weights)

        # Modifie and concat the List of outputs and targets
        # and x-Axis as the output values and the targets as y-Axis
        outputs = [np.transpose(o).tolist()[0] for o in outputs]
        targets = [np.transpose(t).tolist()[0] for t in targets]

        # Set the new Lists
        output_values = [] # x-Points
        target_values = [] # y-Points

        for (o, t) in zip(outputs, targets):
            output_values += o
            target_values += t
        # for

        changing_factor = 1 / float(len(output_values))
        # Add to every value a change factor for better plotting
        for i in xrange(0, len(output_values)):
            output_values[i] += float(i) * changing_factor
            target_values[i] += float(i) * changing_factor
        # for

        xy = []
        for (o, t) in zip(output_values, target_values):
            xy.append((o, t))
        # for

        sorted(xy, key = lambda x : x[0])

        output_values = []
        target_values = []
        for (o, t) in xy:
            output_values.append(o)
            target_values.append(t)
        # for        

        x = np.array([0.0] + output_values)
        y = np.array([0.0] + target_values)

        print(str(output_values))
        print(str(target_values))
        
        x = x[:, np.newaxis]
        a, _, _, _ = np.linalg.lstsq(x, [0.0] + target_values)

        # Now plot the graph
        plt.figure()
        
        plt.plot(output_values, target_values, "ro")
        plt.plot(x, a*x, "r-")
        print("a = "+str(a))

        plt.show(block = False)
    # def show_outputs_targets_plot

    def show_own_outputs_targets_plot_points(self):
        # The the network inputs and targets
        inputs = self.inputs
        targets = self.targets
        self.show_outputs_targets_plot(inputs, targets)
    # def show_outputs_targets_plot_points

    def show_outputs_targets_plot_bars(self):
        inputs = self.inputs
        targets = self.targets
        
    # def show_outputs_targets_plot_points

    def show_plot(self, error_lists):
        print("comes in!")
        for el in error_lists:
            p_x = [i for i in xrange(0, len(el))]
            p_y = [e for e in el]
            plt.plot(p_x, p_y)
        # for

        plt.yscale("log")
        plt.show()

        # p_x = [i for i in range(0, len(errors))]
        # p_y = [float(e) for e in errors]
        # p_x_dif = [i for i in range(0, len(diffs))]
        # p_y_dif = [d for d in diffs]
        # p_x_dif2 = [i for i in range(0, len(diffs2))]
        # p_y_dif2 = [d2 for d2 in diffs2]

        # if self.is_show_plot is True:
        #     plt.figure(0)
            
        #     ax1 = plt.subplot(111)
        #     ax1.plot(p_x, p_y)
        #     plt.yscale('log')
        #     plt.title('Learning curve of the Neural Network')
        #     plt.ylabel('Absolute Error from inputs\nto current outputs and targets')
        #     plt.xlabel('Number of Iterations')
        #     # plt.figure(1)
            
        #     plt.figure(1)
        #     ax2 = plt.subplot(211)
        #     ax2.plot(p_x_dif, p_y_dif)
        #     plt.yscale('log')
        #     # ax2.yscale('log')

        #     # plt.figure(2)
        #     ax3 = plt.subplot(212)
        #     ax3.plot(p_x_dif2, p_y_dif2)
        #     plt.yscale('log')
        #     # ax3.yscale('log')
        #     plt.show()
    # def is_show_plot

	def picture_list_to_matrix(img_1d_array, width, height, scale):
	    w, h = width, height
	    # scale = 20
	    print("scale factor in pucture list to matrix = "+str(scale))

	    data = np.zeros((scale*h, scale*w, 3), dtype=np.uint8)
	        
	    img_2d_array = [list(img_1d_array[w*i:w*(i+1)]) for i in xrange(0, h)]
	    # print("img_2d_array = "+str(img_2d_array))
	    # print("img_array = "+str(img_array))

	    # print("len y = "+str(len(img_2d_array))+"    len x = "+str(len(img_2d_array[0])))
	    # print("img_2d_array =\n"+str(img_2d_array))
	    for y in xrange(0, h):
	        for x in xrange(0, w):
	            # print("x = "+str(x)+"   y = "+str(y))
	            color_value = int(float(img_2d_array[y][x]) * 255.)
	            color = (color_value, color_value, color_value)
	            for ys in xrange(0, scale):
	                for xs in xrange(0, scale):
	                    data[scale*y+ys, scale*x+xs] = color
	                # for
	            # for
	        # for
	    # for

	    return data
	# def pixels_list_to_2d_list

	def show_picture(input_array, value, index, width, height, scale):
	    # w, h = 28, 28
	    # w, h = 14, 14
	    input_array = self.inputs[index]
	    width = math.sqrt(len(input_array))

	    print("scale factor in show = "+str(scale))
	    data = picture_list_to_matrix(img_1d_array, width, height, scale)
	    img = Image.fromarray(data, "RGB")
	    img.show(title="Number "+str(value))
	    # img.save(picture_directory+"train_"+str(index)+"_num_"+str(list_pixels[2][index])+".png")

	    print("The Digit of picture #"+str(index)+" is: "+str(value))
	# def sace_pictures_in_directory

    def get_random_matrix(self, rows, columns, min_val, max_val, roundCom):
        matrix = []

        if self.is_decimal:
            i = 0
            while i < rows:
                tmp = []
                j = 0
                while j < columns:
                    tmp.append(dec(round(uniform(min_val, max_val), roundCom)))
                    j += 1
                matrix.append(tmp)
                i += 1
        else:
            i = 0
            while i < rows:
                tmp = []
                j = 0
                while j < columns:
                    tmp.append(round(uniform(min_val, max_val), roundCom))
                    j += 1
                matrix.append(tmp)
                i += 1
        # if

        return matrix
    # def get_random_matrix

    def get_input_layer(self):
        return self.inputs_amount

    def get_output_layer(self):
        return self.outputs_amount

    def get_neuron_list(self):
        return self.neuron_list

    def get_random_biases_list(self, neuron_list, nmin = -3, nmax = 3, ncommas = 5):
        nl = neuron_list
        biases = [np.array(self.get_random_matrix(1, nl[i + 1], nmin, nmax, ncommas)) for i in xrange(0, len(nl) - 1)]
        return biases

    def get_random_weights_list(self, neuron_list, nmin = -3, nmax = 3, ncommas = 5):
        nl = neuron_list
        weights = [np.array(self.get_random_matrix(nl[i], nl[i + 1], nmin, nmax, ncommas)) for i in xrange(0, len(nl) - 1)]
        return weights

    def get_zero_biases_weights(self):
        biases = [np.zeros((1, i)) for i in self.neuronal_list[1:]]
        weights = [np.zeros((i, j)) for (i, j) in zip(self.neuronal_list[:-1], self.neuronal_list[1:])]
        
        return biases, weights
    # def get_zero_biases_weights

    def get_random_neurons_weights(self, neuron_list, nmin = -3, nmax = 3, ncommas = 5):
        nl = neuron_list
        biases = [np.array(self.get_random_matrix(1, nl[i + 1], nmin, nmax, ncommas)) for i in xrange(0, len(nl) - 1)]
        weights = [np.array(self.get_random_matrix(nl[i], nl[i + 1], nmin, nmax, ncommas)) for i in xrange(0, len(nl) - 1)]
        return biases, weights

    def get_network_error(self, network, inputs, targets):
        nl, bs, ws = network[0], network[1], network[2]

        outl = [self.calculate_forward(i, bs, ws) for i in inputs]

        errl = [self.square_error(out, t) for out, t in zip(outl, targets)]

        error_total = 0
        for e in errl:
            error_total += e

        return error_total
    # def get_network_error

    # Setter
    def set_neuron_list(self, inputs):# for input, hidden and output layer simultaneously
        self.matrix_init = False
        self.set_input_layer(inputs[0])
        self.set_hidden_layer(inputs[1:-1])
        self.set_output_layer(inputs[-1])
        self.neuron_list = inputs

    def set_input_layer(self, inputs):
        self.matrix_init = False
        self.inputs_amount = inputs

    def set_output_layer(self, outputs):
        self.matrix_init = False
        self.outputs_amount = outputs

    def set_hidden_layer(self, numbers):
        if type(numbers) != type([]):
            print("Must be type of lists! Hidden layer was not changed!")
        # if
        if len(numbers) < 0:
            print("No negative number possible!")
        elif len(numbers) > 10:
            print("No more than 10 Layer possible! Set to 10 hidden Layers")
            self.hidden = numbers[0:10]
        else:
            print("Set amount of hidden layer to " + str(len(numbers)))
            self.hidden = numbers
        # if
    # def set_hidden_layer

    def get_network(self):
        return [self.neuron_list, self.biases, self.weights]
    # def get_network

    def set_network(self, network):
        self.neuronal_list = network[0]
        self.biases = network[1]
        self.weights = network[2]
    # def set_network

    def init_random_weights(self):
        # Create a network with hidden layer
        nl = self.neuron_list
        self.weights = []
        self.biases = []
        for i in range(0, len(nl) - 1):
            self.weights.append(np.array(self.get_random_matrix(nl[i], nl[i + 1], -3, 3, 1)))
            self.biases.append(np.array(self.get_random_matrix(1, nl[i + 1], -3, 3, 1)))


        self.reset_saved_networks()
        self.weights_init = True
        print("Created random biases and weights for "+str(self.neuron_list)+" neuronal list")
    # def init_random_weights

    def init_random_targets(self):
        if not self.weights_init:
            self.init_random_weights()
        # if

        if self.targets_amount_set:
            targets_amount = self.targets_amount
        else:
            targets_amount = randint(1, 100)
        # if

        nl = self.neuron_list
        ta = self.targets_amount
        self.inputs = [np.array(self.get_random_matrix(nl[0], 1, -3, 3, 3))
                  for _ in range(0, ta)]
        self.targets = [np.array(self.get_random_matrix(nl[-1], 1, 0.1, 0.9, 3))
                   for _ in range(0, ta)]

        self.targets_init = True
        print("Create "+str(self.targets_amount)+" inputs and targets")
    # def init_random_targets

    # Calculate a given network through
    # param weights: Weights of the neurons with bias
    # param input: Input of the first Layer of the Network
    def calculate_forward(self, start_input, biases, weights):
        if self.is_decimal:
            _sigmoid = self.sigmoid_decimal
        else:
            _sigmoid = self.sigmoid
        # if

        x = start_input
        for i in range(0, len(weights)):
            b = biases[i]
            w = weights[i]
            # b = np.outer(biases[i], np.ones(len(w[0])))
            wi = np.array(list(b) + list(w))
            x = np.array([[1]] + list(x))
            # print("wi =\n"+str(wi)+"\nx =\n"+str(x))
            x = np.dot(wi.transpose(), x)
            x = _sigmoid(x)
        # for

        return x
    # def calculate_forward

    def calculate_forward_many(self, inputs, biases, weights):
        outputs = []
        
        for inp in inputs:
            outputs.append(self.calculate_forward(inp, biases, weights))
        # for

        return outputs
    # def calculate_forward_many

    def backpropagation(self, start_input, biases, weights, targets, etha):
        # Feed forward
        xs = []
        ys = [start_input]

        # Set all functions for calculations
        if self.is_decimal:
            _sigmoid = self.sigmoid_decimal
            _sigmoid_invert = self.sigmoid_invert_decimal
            _scalar_matrix_multiplication = self.scalar_matrix_multiplication_decimal
        else:
            _sigmoid = self.sigmoid
            _sigmoid_invert = self.sigmoid_invert
            _scalar_matrix_multiplication = self.scalar_matrix_multiplication
        # if

        y = start_input
        for i in xrange(0, len(weights)):
            x = y
            b = biases[i]
            w = weights[i]
            # b = np.outer(biases[i], np.ones(len(w[0])))
            wi = np.array(list(b) + list(w))
            x = np.array([[1]] + list(x))
            a = np.dot(wi.transpose(), x)
            xs.append(a)
            y = _sigmoid(a)
            ys.append(y)
        # for

        # Backward error correction
        bs = deepcopy(biases)
        ws = deepcopy(weights)

        d = np.subtract(ys[-1], targets) * _sigmoid_invert(xs[-1])
        bs[-1] = d.transpose()

        ws[-1] = np.dot(d, ys[-2].transpose())

        for i in xrange(2, len(weights) + 1):
            x = xs[-i]
            xp = _sigmoid_invert(x)

            d = np.dot(weights[-i + 1], d) * xp

            bs[-i] = d.transpose()
            ws[-i] = np.dot(d, ys[-i-1].transpose()).transpose()
        # for

        bsd = [_scalar_matrix_multiplication(etha, bs[i]) for i in xrange(0, len(biases))]
        ws[-1] = ws[-1].transpose()
        wsd = [_scalar_matrix_multiplication(etha, ws[i]) for i in xrange(0, len(weights))]

        return (bsd, wsd)
    # def backpropagation

    def backprop_many(self, inputs, biases, weights, targets, etha, finish_calc=False):
        bsdt = []
        wsdt = []
        for i in xrange(0, len(inputs)):
            bsd, wsd = self.backpropagation(inputs[i], biases, weights, targets[i], etha)
            bsdt.append(bsd)
            wsdt.append(wsd)
        # for

        # TODO: add a function, where the difference will be calculated at once!

        return bsdt, wsdt
    # def backprop_many    

    def get_improved_network(self, network, bsdt, wsdt, copy=False):
        if copy:
            network = deepcopy(network)
        # if

        nl, bs, ws = network[0], network[1], network[2]
        range_nl = xrange(0, len(nl) - 1)
        range_bsdl = xrange(0, len(bsdt))

        for k in range_nl:
            for j in range_bsdl:
                bs[k] = np.subtract(bs[k], bsdt[j][k])
                ws[k] = np.subtract(ws[k], wsdt[j][k])
            # for
        # for

        return [nl, bs, ws]
    # def get_improved_network

    def get_improved_network_components(self, nl, bs, ws, bsdt, wsdt, copy=False):
        if copy:
            nl, bs, ws = deepcopy(nl), deepcopy(bs), deepcopy(ws)
        # if

        # nl, bs, ws = network[0], network[1], network[2]
        range_nl = xrange(0, len(nl) - 1)
        range_bsdl = xrange(0, len(bsdt))

        for k in range_nl:
            for j in range_bsdl:
                bs[k] = np.subtract(bs[k], bsdt[j][k])
                ws[k] = np.subtract(ws[k], wsdt[j][k])
            # for
        # for

        return [nl, bs, ws]
    # def get_improved_network

    # Improve network offline
    def improve_network(self, is_bsd, network, inputs, targets, iterations, etha, copy = False):
        if copy:
            network = deepcopy(network)
        # if

        if self.is_decimal:
            etha = dec(etha)
            last_layer = dec(network[0][len(network[0]) - 1])
            len_inputs = dec(len(inputs))
            _square_error_many = self.square_error_many_decimal
        else:
            etha = float(etha)
            last_layer = network[0][len(network[0]) - 1]
            len_inputs = len(inputs)
            _square_error_many = self.square_error_many
        # if

        range_iterations = xrange(0, iterations)
        range_inputs = xrange(0, len(inputs))

        error_list = []

        # First calc the current absolute error of network with inputs and targets
        targets_calc = self.calculate_forward_many(inputs, network[1], network[2])
        error_list.append(_square_error_many(targets_calc, targets) / last_layer / len_inputs)

        network_copy = deepcopy(network)
        for itr in range_iterations:
            if is_bsd:
                # Then Improve it with bgd
                bsdt, wsdt = self.backprop_many(inputs, network[1], network[2], targets, etha)
                network = self.get_improved_network(network, bsdt, wsdt)

                # And also calc the error
                # targets_calc = self.calculate_forward(inputs, network[1], network[2])
                # error_list.append(_square_error_many(targets_calc, targets) / last_layer / len_inputs)
            else:
                # Then Improve it with sgd
                for i in range_inputs:
                    bsd, wsd = self.backpropagation(inputs[i], network[1], network[2], targets[i], etha)
                    network = self.get_improved_network(network, [bsd], [wsd])
                # for
            #if

            targets_calc = self.calculate_forward_many(inputs, network[1], network[2])
            error_list.append(_square_error_many(targets_calc, targets) / last_layer / len_inputs)
            
            stdoutw(".")
            if itr > 0 and ((itr+1) % 64) == 0: stdoutw("\n")
        # for
        stdoutw("\n")

        # if copy:
        #     error_list = deepcopy(error_list)
        # # if
        return network, error_list
    # def improve_network_sgd

    def improve_network_itself(self, is_bsd, inputs, targets, iterations, copy = False):
        if self.matrix_init is False:
            print(self.error_not_init)

        network = [self.neuron_list, self.biases, self.weights]
        if copy:
            network = deepcopy(network)
        # if

        if is_bsd:
            network, error_list = self.improve_network(True, network, inputs, targets, iterations, self.learning_rate)
        else:
            network, error_list = self.improve_network(False, network, inputs, targets, iterations, self.learning_rate)
        # if
        
        self.biases = deepcopy(network[1])
        self.weights = deepcopy(network[2])
        self.error_list = deepcopy(error_list)

        self.last_error = error_list[-1]
        print("last error was: "+str(self.last_error))
        if self.is_prev_error_there == True:
            print("prev last error was: "+str(self.prev_error))
            print("diff of last and prev last: "+str(self.last_error - self.prev_error))
        else:
            self.is_prev_error_there = True
        # if
        self.prev_error = self.last_error
        return network, error_list
    # def calculate_with_initial_values_sgd

    def bgd(self, iterations, copy = False):
        return self.improve_network_itself(True, self.inputs, self.targets, iterations, copy)
    # def bgd

    def sgd(self, iterations, copy = False):
        return self.improve_network_itself(False, self.inputs, self.targets, iterations, copy)
    # def

    # Random Part, is for testing the Neural Network
    # Is used for multi Processing, because multi threading is in Python
    # really really slow!
    def get_delta_biases_weights(self, queue, is_finished, mutex, proc_num, network, inputs, targets, etha):#, show_debug_msg = False):
        nl, bs, ws = network.neuronal_list, network.biases, network.weights
        testing_amount = len(inputs)
        print("testing amount is "+str(testing_amount))
        divide_factor = testing_amount * nl[-1]

        print("start proc num #"+str(proc_num))

        # Do this step by step
        error_total = 0

        bsdt, wsdt = self.backpropagation(inputs[0], bs, ws, targets[0], etha)
        output = self.calculate_forward(inputs[0], bs, ws)
        error_total += self.square_error(output, targets[0])
        print("proc # is "+str(proc_num)+", j = 0")
        
        for j in range(1, testing_amount):
            bsd, wsd = self.backpropagation(inputs[j], bs, ws, targets[j], etha)
            for i in xrange(0, len(bsd)):
                bsdt[i] += bsd[i]
                wsdt[i] += wsd[i]
            # for
            output = self.calculate_forward(inputs[j], bs, ws)
            error_total += self.square_error(output, targets[j])
            print("proc # is "+str(proc_num)+", j = "+str(j))
        # for

        error_total /= divide_factor

        print("proc #"+str(proc_num)+" finished with calculation!")

        # Lock for thread / process safe
        # mutex.aquire()
        with mutex:
            queue.put((bsdt, wsdt, error_total))
            is_finished.put(proc_num)
        
        # Release to not bget deadlock
        # mutex.release()
    # def get_delta_biases_weigths

    def get_delta_biases_weights_multiprocess(self, queue_result, queue_is_finished, queue_output, mutex, mutex_print, proc_num, network, inputs, targets, etha):#, show_debug_msg = False):
        nl, bs, ws = network.neuronal_list, network.biases, network.weights
        bs = deepcopy(bs) # make a copy of it, to can learn it in this function!
        ws = deepcopy(ws)
        error_list = []
        testing_amount = len(inputs)

        # mutex_print.acquire()
        print("testing amount is "+str(testing_amount))
        # mutex_print.release()

        divide_factor = testing_amount * nl[-1]

        # mutex_print.acquire()
        print("start proc num #"+str(proc_num))
        # mutex_print.release()

        # Do this step by step
        error_total = 0

        bsdt, wsdt = self.backpropagation(inputs[0], bs, ws, targets[0], etha)
        output = self.calculate_forward(inputs[0], bs, ws)
        error_total += self.square_error(output, targets[0])

        # mutex_print.acquire()
        # print("proc # is "+str(proc_num)+", j = 0")
        # mutex_print.release()
        
        for j in range(1, testing_amount):
            bsd, wsd = self.backpropagation(inputs[j], bs, ws, targets[j], etha)
            for i in xrange(0, len(bsd)):
                bs[i] -= bsd[i] # learn the network
                ws[i] -= wsd[i]
                bsdt[i] += bsd[i] # and also save the total difference of this funciton
                wsdt[i] += wsd[i]
            # for
            output = self.calculate_forward(inputs[j], bs, ws)
            error_local = self.square_error(output, targets[j])

            error_list.append(error_local)
            error_total += error_local

            queue_output.put(str(proc_num))
        # for

        error_total /= divide_factor

        print("proc #"+str(proc_num)+" finished with calculation!")

        mutex.acquire()
        queue_result.put((bsdt, wsdt, error_list, error_total))
        queue_is_finished.put(proc_num)
        mutex.release()
    # def get_delta_biases_weigths

    def get_approximation_guess(self, queue_result, is_finished, mutex, proc_num, network, inputs, real_values, start_pos):
        nl, bs, ws = network.neuronal_list, network.biases, network.weights
        outputs = self.calculate_forward_many(inputs, bs, ws)

        outputs = [o.transpose().tolist()[0] for o in outputs]
        # print(str(outputs))

        result = []
        for i in xrange(0, len(outputs)):
            temp = []
            for j in xrange(0, len(outputs[i])):
                temp.append([j, outputs[i][j]])
            temp = sorted(temp, key=itemgetter(1), reverse=True)

            result.append((i+start_pos, real_values[i], temp))
        # for

        print("proc #"+str(proc_num)+" finished with calculation!")

        mutex.acquire()
        queue_result.put(result)
        is_finished.put(proc_num)
        mutex.release()
       	# with
    # def get_approxiamtion_guess

    def test_function(self):
    	# print("This should be the 1st neural network DrawNumbers function!")
    	print("This is the 33333rd      DrawNumbers function for other neural network!")
    # def test_function
# class NeuralNetwork
