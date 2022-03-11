#! /usr/bin/python2.7

# 2015.07.13
# ipython
# klassen erstellen
# stochastic gratienten descent (abstieg)
# batch GD
# use python 2.7!

import sys
import os

import numpy as np
import Tkinter as tk
import matplotlib.pyplot as plt

from random import randint
from random import uniform

from copy import deepcopy
from operator import itemgetter

# For colorful output in Terminal
import colorama
from colorama import Fore, Back, Style

# For clearing the Terminal
def cls():
    os.system('cls' if os.name == 'nt' else 'clear')

# Helpful for getting the max depth in a list
depth = lambda L: isinstance(L, list) and max(map(depth, L))+1

__author__ = "haris"
__date__ = "$Jun 24, 2015 11:10:57 AM$"

## Class for Network
## Class for Weights
## Class for Biases

class NeuralNetwork(Exception):

    error_not_init = "Please, initial first the matrices!\nUse e.g. init_random() or other functions"

    def scalar_matrix_multiplication(self, scalar, matrix):
        scalar_matrix = [[scalar for i2 in range(0, len(list(matrix[0])))] for i1 in range(0, len(list(matrix)))]
        scalar_matrix = np.array(scalar_matrix)

        return scalar_matrix * matrix

    def f(self, a) : return (1 / (1 + np.exp(-a)))

    def fi(self, a) : return (self.f(a) * self.f(1 - a))

    def sigmoid(self, vector):
        new_vector_list = [val.tolist()[0] for val in vector]
        return np.array([[self.f(v)] for v in new_vector_list])

    def sigmoid_invert(self, vector):
        new_vector_list = [val.tolist()[0] for val in vector]
        return np.array([[self.fi(v)] for v in new_vector_list])

    def square_error(self, v1, v2):
        square_error = 0
        for n1, n2 in zip(v1.transpose().tolist()[0], v2.transpose().tolist()[0]):
            square_error += (float(n1) - float(n2))**2
        return square_error
    # def square_error

    def square_error_targets(self, targets_calc, targets):
        square_error = 0
        for tc, t in zip(targets_calc, targets):
            square_error += self.square_error(tc, t)
        # for
        return square_error
    # def square_error_of_targets

    # # Sigmoid Function
    # def f(self, a):
    #     return 1 / (1 + np.e**(-a))
    # # Sigmoid Derivative
    # def fi(self, a):
    #     return self.f(a) * self.f(1-a)

    def __init__(self, inputs = 1, outputs = 1, targets = 5):
        self.inputs_amount = inputs
        self.outputs_amount = outputs
        self.hidden = []
        self.matrix_init = False
        self.weights_init = False
        self.targets_init = False
        self.neuron_list = [inputs, outputs]
        self.weights = []
        self.biases = []
        self.error_list = []
        self.targets_amount = targets
        if (targets > 0):
            self.targets_amount_set = True
        else:
            self.targets_amount_set = False
        # if

        self.io_targets = []
        self.learning_rate = 0.001
        self.is_show_plot = True

    def stats(self):
        print("inputs = " + str(self.inputs_amount))
        print("outputs = " + str(self.outputs_amount))
        if len(self.hidden) == 0:
            print("No hidden layer")
        else:
            print(str(len(self.hidden)) + " layers set with " + str(self.hidden) + " neurons")
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
 
        print("Plot is set to "+str(self.is_show_plot))

    def show_error_plot(self):
        p_x = [i for i in xrange(0, len(self.error_list))]
        p_y = [e for e in self.error_list]
        plt.plot(p_x, p_y)

        plt.yscale("log")
        plt.show(block = False)
    # def show_error_plot

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

    def get_random_matrix(self, rows, columns, min, max, roundCom):
        matrix = []

        i = 0
        while i < rows:
            tmp = []
            j = 0
            while j < columns:
                tmp.append(round(uniform(min, max), roundCom))
                j += 1
            matrix.append(tmp)
            i += 1

        return matrix
    # def get_random_matrix

    def get_input_layer(self):
        return self.inputs_amount

    def get_output_layer(self):
        return self.outputs_amount

    def get_neuron_list(self):
        return self.neuron_list

    def get_random_biases_list(self, neuron_list, nmin = -3, nmax = 3, ncommas = 3):
        nl = neuron_list
        biases = [np.array(self.get_random_matrix(1, nl[i + 1], nmin, nmax, ncommas)) for i in xrange(0, len(nl) - 1)]
        return biases

    def get_random_weight_list(self, neuron_list, nmin = -3, nmax = 3, ncommas = 3):
        nl = neuron_list
        weights = [np.array(self.get_random_matrix(nl[i], nl[i + 1], nmin, nmax, ncommas)) for i in xrange(0, len(nl) - 1)]
        return weights

    def get_random_neurons_weights(self, neuron_list, nmin = -3, nmax = 3, ncommas = 3):
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
        if len(numbers) < 0:
            print("No negative number possible!")
        elif len(numbers) > 10:
            print("No more than 10 Layer possible! Set to 10 hidden Layers")
            self.hidden = numbers[0:10]
        else:
            print("Set amount of hidden layer to " + str(len(numbers)))
            self.hidden = numbers

    def init_random_weights(self):
        # Create a network with hidden layer
        nl = self.neuron_list
        self.weights = []
        self.biases = []
        for i in range(0, len(nl) - 1):
            self.weights.append(np.array(self.get_random_matrix(nl[i], nl[i + 1], -3, 3, 1)))
            self.biases.append(np.array(self.get_random_matrix(1, nl[i + 1], -3, 3, 1)))

        self.weights_init = True
        print("Created random biases and weights for "+str(self.neuron_list)+" neural list")
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
    def calculate_forward(self, inputs, biases, weights):
        targets = []
        for inp in inputs:
            x = inp
            for i in range(0, len(weights)):
                b = biases[i]
                w = weights[i]
                # b = np.outer(biases[i], np.ones(len(w[0])))
                wi = np.array(list(b) + list(w))
                x = np.array([[1]] + list(x))
                x = np.dot(wi.transpose(), x)
                x = self.sigmoid(x)
            # for
            targets.append(x)
        # for
        return targets
    # def calculate_forward

    def backpropagation(self, start_input, biases, weights, targets, etha):
        # Feed forward
        xs = []
        ys = [start_input]

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
            y = self.sigmoid(a)
            ys.append(y)
        # for

        # Backward error correction
        bs = deepcopy(biases)
        ws = deepcopy(weights)

        d = np.subtract(ys[-1], targets) * self.sigmoid_invert(xs[-1])
        bs[-1] = d.transpose()

        ws[-1] = np.dot(d, ys[-2].transpose())

        for i in xrange(2, len(weights) + 1):
            x = xs[-i]
            xp = self.sigmoid_invert(x)

            d = np.dot(weights[-i + 1], d) * xp

            bs[-i] = d.transpose()
            ws[-i] = np.dot(d, ys[-i-1].transpose()).transpose()
        # for

        bsd = [self.scalar_matrix_multiplication(etha, bs[i]) for i in xrange(0, len(biases))]

        ws[-1] = ws[-1].transpose()

        wsd = [self.scalar_matrix_multiplication(etha, ws[i]) for i in xrange(0, len(weights))]
        return (bs, ws)
    # def backpropagation

    def backprop_many(self, inputs, biases, weights, targets, etha):
        bsdl = []
        wsdl = []
        for i in xrange(0, len(inputs)):
            bsd, wsd = self.backpropagation(inputs[i], biases, weights, targets[i], etha)
            bsdl.append(bsd)
            wsdl.append(wsd)
        # for

        return bsdl, wsdl
    # def backprop_many    

    # Improve network online
    def improve_network_sgd(self, network, inputs, targets, iterations, etha):
        error_list = []
        last_layer = network[0][len(network[0]) - 1]
        len_inputs = len(inputs)
        network = deepcopy(network)
        range_iterations = xrange(0, iterations)
        range_inputs = xrange(0, len(inputs))

        # First calc the current absolute error of network with inputs and targets
        targets_calc = self.calculate_forward(inputs, network[1], network[2])
        error_list.append(self.square_error_targets(targets_calc, targets) / last_layer / float(len_inputs))

        for itr in range_iterations:
            # Then Improve it
            for i in range_inputs:
                bsd, wsd = self.backpropagation(inputs[i], network[1], network[2], targets[i], etha)
                network = self.get_improved_network(network, [bsd], [wsd])
            # for
            # And also calc the error
            targets_calc = self.calculate_forward(inputs, network[1], network[2])
            error_list.append(self.square_error_targets(targets_calc, targets) / last_layer / float(len_inputs))
        # for

        # self.error_list = deepcopy(error_list)
        return network, error_list
    # def improve_network_sgd

    # Improve network offline
    def improve_network_bgd(self, network, inputs, targets, iterations, etha):
        error_list = []
        last_layer = network[0][len(network[0]) - 1]
        len_inputs = len(inputs)
        network = deepcopy(network)
        range_iterations = xrange(0, iterations)

        # First calc the current absolute error of network with inputs and targets
        targets_calc = self.calculate_forward(inputs, network[1], network[2])
        error_list.append(self.square_error_targets(targets_calc, targets) / last_layer / float(len_inputs))

        for itr in range_iterations:
            # Then Improve it
            bsdl, wsdl = self.backprop_many(inputs, network[1], network[2], targets, etha)
            network = self.get_improved_network(network, bsdl, wsdl)
            # And also calc the error
            targets_calc = self.calculate_forward(inputs, network[1], network[2])
            error_list.append(self.square_error_targets(targets_calc, targets) / last_layer / float(len_inputs))
        # for

        # self.error_list = deepcopy(error_list)
        return network, error_list
    # def improve_network_sgd

    def calculate_with_initial_values_sgd(self, inputs, targets, iterations):
        # if self.matrix_init is False:
        #     print(self.error_not_init)
        network = [self.neuron_list, self.biases, self.weights]
        network, error_list = self.improve_network_sgd(network, inputs, targets, iterations, self.learning_rate)

        self.biases = deepcopy(network[1])
        self.weights = deepcopy(network[2])
        self.error_list = deepcopy(error_list)
        return network, error_list
    # def calculate_with_initial_values_sgd

    def calculate_with_initial_values_bgd(self, inputs, targets, iterations):
        # if self.matrix_init is False:
        #     print(self.error_not_init)
        network = [self.neuron_list, self.biases, self.weights]
        network, error_list = self.improve_network_bgd(network, inputs, targets, iterations, self.learning_rate)
        
        self.biases = deepcopy(network[1])
        self.weights = deepcopy(network[2])
        self.error_list = deepcopy(error_list)
        return network, error_list
    # def calculate_with_initial_values_bgd

    def get_improved_network(self, network, bsdl, wsdl):
        network = deepcopy(network)
        nl, bs, ws = network[0], network[1], network[2]
        range_nl = xrange(0, len(nl) - 1)
        range_bsdl = xrange(0, len(bsdl))

        for k in range_nl:
            for j in range_bsdl:
                bs[k] = np.subtract(bs[k], bsdl[j][k])
                ws[k] = np.subtract(ws[k], wsdl[j][k])
            # for
        # for

        return [nl, bs, ws]
    # def get_improved_network

    # Random Part, is for testing the Neural Network
    # Is used for multi Processing, because multi threading is in Python
    # really really slow!
    def get_delta_biases_weights(self, queue, is_finished, proc_num, network, inputs, targets, etha):#, show_debug_msg = False):
        nl, bs, ws = network[0], network[1], network[2]

        testing_amount = len(inputs)

        outputs = [self.calculate_forward(inputs[j], bs, ws) for j in range(0, testing_amount)]

        errors_temp = [self.square_error(outputs[j], targets[j]) for j in range(0, testing_amount)]
        error_total = 0
        for j in range(0, testing_amount):
            error_total += errors_temp[j]

        divide_factor = testing_amount * nl[-1]
        error_total /= divide_factor
        bsd = [0 for j in range(0, testing_amount)]
        wsd = [0 for j in range(0, testing_amount)]

        for j in range(0, testing_amount):
            bsd[j], wsd[j] = self.backpropagation(inputs[j], bs, ws, targets[j], etha)
            print("proc # is "+str(proc_num)+", j = "+str(j))

        outputs = [self.calculate_forward(inputs[j], bs, ws) for j in range(0, testing_amount)]
        errors_temp = [self.square_error(outputs[j], targets[j]) for j in range(0, testing_amount)]
        error_total = 0
        for j in range(0, testing_amount):
            error_total += errors_temp[j]

        error_total /= divide_factor

        print("proc #"+str(proc_num)+" finished with calculation!")

        queue.put((bsd, wsd, error_total))
        is_finished.put(proc_num)
    # def get_delta_biases_weigths

    def get_approximation_guess(self, queue, is_finished, proc_num, network, inputs, real_values, start_pos):
        nl, bs, ws = network[0], network[1], network[2]
        outputs = [self.calculate_forward(i, bs, ws) for i in inputs]

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
        queue.put(result)
        is_finished.put(proc_num)
    # def get_approxiamtion_guess
# class NeuralNetwork
