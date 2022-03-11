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

    # # Sigmoid Function
    # def f(self, a):
    #     return 1 / (1 + np.e**(-a))
    # # Sigmoid Derivative
    # def fi(self, a):
    #     return self.f(a) * self.f(1-a)

    def __init__(self, inputs = 1, outputs = 1):
        self.inputs = inputs
        self.outputs = outputs
        self.hidden = []
        self.matrix_init = False
        self.weights_init = False
        self.targets_init = False
        self.neuron_list = [inputs, outputs]
        self.weights = []
        self.biases = []
        self.targets_amount = 0
        self.io_targets = []
        self.learning_rate = 0.005
        self.show_plot = True

    def stats(self):
        print("inputs = " + str(self.inputs))
        print("outputs = " + str(self.outputs))
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
 
        print("Plot is set to "+str(self.show_plot))

    def __get_random_matrix(self, rows, columns, min, max, roundCom):
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

    # Getter
    def get_input_layer(self):
        return self.inputs

    def get_output_layer(self):
        return self.outputs

    def get_neuron_list(self):
        return self.neuron_list

    def get_random_biases_list(self, neuron_list, nmin = -3, nmax = 3, ncommas = 3):
        nl = neuron_list
        biases = [np.array(self.__get_random_matrix(1, nl[i + 1], nmin, nmax, ncommas)) for i in xrange(0, len(nl) - 1)]
        return biases

    def get_random_weight_list(self, neuron_list, nmin = -3, nmax = 3, ncommas = 3):
        nl = neuron_list
        weights = [np.array(self.__get_random_matrix(nl[i], nl[i + 1], nmin, nmax, ncommas)) for i in xrange(0, len(nl) - 1)]
        return weights

    def get_random_neurons_weights(self, neuron_list, nmin = -3, nmax = 3, ncommas = 3):
        nl = neuron_list
        biases = [np.array(self.__get_random_matrix(1, nl[i + 1], nmin, nmax, ncommas)) for i in xrange(0, len(nl) - 1)]
        weights = [np.array(self.__get_random_matrix(nl[i], nl[i + 1], nmin, nmax, ncommas)) for i in xrange(0, len(nl) - 1)]
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
        self.inputs = inputs

    def set_output_layer(self, outputs):
        self.matrix_init = False
        self.outputs = outputs

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
            self.weights.append(np.array(self.__get_random_matrix(nl[i], nl[i + 1], -3, 3, 1)))
            self.biases.append(np.array(self.__get_random_matrix(1, nl[i + 1], -3, 3, 1)))

        self.weights_init = True
        print("Created random biases and weights for "+str(self.neuron_list)+" neural list")

    def init_random_targets(self):
        self.targets_amount = randint(1, 100)        
        nl = self.neuron_list
        ta = self.targets_amount
        inputs = [np.array(self.__get_random_matrix(nl[0], 1, -3, 3, 3))
                  for _ in range(0, ta)]
        targets = [np.array(self.__get_random_matrix(nl[-1], 1, 0.1, 0.9, 3))
                   for _ in range(0, ta)]

        self.targets_init = True
        print("Create "+str(self.targets_amount+" inputs and targets"))

    # Calculate a given network through
    # param weights: Weights of the neurons with bias
    # param input: Input of the first Layer of the Network
    def calculate_forward(self, ninput, biases, weights):
        x = ninput
        for i in range(0, len(weights)):
            b = biases[i]
            w = weights[i]
            # b = np.outer(biases[i], np.ones(len(w[0])))
            wi = np.array(list(b) + list(w))
            x = np.array([[1]] + list(x))
            x = np.dot(wi.transpose(), x)
            x = self.sigmoid(x)
        return x

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

        bs = [self.scalar_matrix_multiplication(etha, bs[i]) for i in xrange(0, len(biases))]

        ws[-1] = ws[-1].transpose()

        ws = [self.scalar_matrix_multiplication(etha, ws[i]) for i in xrange(0, len(weights))]
        return (bs, ws)

    def calculate_with_initial_values_sgd(self):
        if self.matrix_init is False:
            print(self.error_not_init)

    def calculate_with_initial_values_bgd(self):
        if self.matrix_init is False:
            print(self.error_not_init)

    # Random Part, is for testing the Neural Network
    def improve_network(self, network, inputs, targets, etha, iterations, show_plot = False, show_debug_msg = False):
        nl, bs, ws = network[0], network[1], network[2]

        testing_amount = len(inputs)

        if show_debug_msg is True:
            print("Creating "+str(testing_amount)+" inputs and targets...")

        outputs = [self.calculate_forward(inputs[j], bs, ws) for j in range(0, testing_amount)]

        errors_temp = [self.square_error(outputs[j], targets[j]) for j in range(0, testing_amount)]
        error_total = 0
        for j in range(0, testing_amount):
            error_total += errors_temp[j]

        divide_factor = testing_amount * nl[-1]
        error_total /= divide_factor
        errors = [error_total]
        diffs = []
        diffs2 = []
        last_error = error_total
        last2_error = error_total
        bsd = [0 for j in range(0, testing_amount)]
        wsd = [0 for j in range(0, testing_amount)]

        for i in range(0, iterations):
            if i % 1000 == 0 and show_debug_msg is True:
                print("i = "+str(i)+"     error_total = "+str(error_total)+"    error_diff = "+str(last_error-error_total)+"   error diff2 = "+str((last2_error-last_error)-(last_error-error_total)))

            for j in range(0, testing_amount):
                bsd[j], wsd[j] = self.backpropagation(inputs[j], bs, ws, targets[j], etha)

            for j in range(0, testing_amount):
                for k in range(0, len(nl) - 1):
                    bs[k] = np.subtract(bs[k], bsd[j][k])
                    ws[k] = np.subtract(ws[k], wsd[j][k])

            outputs = [self.calculate_forward(inputs[j], bs, ws) for j in range(0, testing_amount)]
            errors_temp = [self.square_error(outputs[j], targets[j]) for j in range(0, testing_amount)]

            last2_error = last_error
            last_error = error_total
            error_total = 0
            for j in range(0, testing_amount):
                error_total += errors_temp[j]
            error_total /= divide_factor
            errors.append(error_total)
            diffs.append(last_error-error_total)
            diffs2.append((last2_error-last_error)-(last_error-error_total))
        # print("last error_total in improve_network is "+str(errors[-1]))

        if self.show_plot is True and show_plot is True:
            p_x = [i for i in range(0, len(errors))]
            p_y = [float(e) for e in errors]
            p_x_dif = [i for i in range(0, len(diffs))]
            p_y_dif = [d for d in diffs]
            p_x_dif2 = [i for i in range(0, len(diffs2))]
            p_y_dif2 = [d2 for d2 in diffs2]

            plt.figure(0)
            
            ax1 = plt.subplot(111)
            ax1.plot(p_x, p_y)
            plt.yscale('log')
            plt.title('Learning curve of the Neural Network')
            plt.ylabel('Absolute Error from inputs\nto current outputs and targets')
            plt.xlabel('Number of Iterations')
            # plt.figure(1)
            
            plt.figure(1)
            ax2 = plt.subplot(211)
            ax2.plot(p_x_dif, p_y_dif)
            plt.yscale('log')
            # ax2.yscale('log')

            # plt.figure(2)
            ax3 = plt.subplot(212)
            ax3.plot(p_x_dif2, p_y_dif2)
            plt.yscale('log')
            # ax3.yscale('log')
            plt.show()

        return nl, bs, ws

    def iterate_with_sgd(self, neuron_list, etha, testing_amount, iterations):
        self.__iterate_with_random_generated_values(neuron_list, etha, testing_amount, iterations, True)

    def iterate_with_bgd(self, neuron_list, etha, testing_amount, iterations):
        self.__iterate_with_random_generated_values(neuron_list, etha, testing_amount, iterations, False)

    def __iterate_with_random_generated_values(self, neuron_list, etha, testing_amount, iterations, is_sgd):
        nl = neuron_list

        print("Creating "+str(testing_amount)+" inputs and targets...")

        inputs = [np.array(self.__get_random_matrix(nl[0],1,-3,3,3)) for i in range(0, testing_amount)]
        if is_sgd is True:
            bref, wref = self.get_random_neurons_weights(nl)
            targets = [self.calculate_forward(inputs[i], bref, wref) for i in range(0, testing_amount)]
        else:
            targets = [np.array(self.__get_random_matrix(nl[-1],1,0.1,0.9,3)) for i in range(0, testing_amount)]

        bs, ws = self.get_random_neurons_weights(nl)
        outputs = [self.calculate_forward(inputs[j], bs, ws) for j in range(0, testing_amount)]

        errors_temp = [self.square_error(outputs[j], targets[j]) for j in range(0, testing_amount)]
        error_total = 0
        for j in range(0, testing_amount):
            error_total += errors_temp[j]

        divide_factor = testing_amount * neuron_list[-1]
        error_total /= divide_factor
        errors = [error_total]
        diffs = []
        diffs2 = []
        last_error = error_total
        last2_error = error_total
        bsd = [0 for j in range(0, testing_amount)]
        wsd = [0 for j in range(0, testing_amount)]

        for i in range(0, iterations):
            if i % 1000 == 0:
                print("i = "+str(i)+"     error_total = "+str(error_total)+"    error_diff = "+str(last_error-error_total)+"   error diff2 = "+str((last2_error-last_error)-(last_error-error_total)))

            if is_sgd is True:
                inputs = [np.array(self.__get_random_matrix(nl[0],1,-3,3,3)) for j in range(0, testing_amount)]
                targets = [self.calculate_forward(inputs[j], bref, wref) for j in range(0, testing_amount)]

            for j in range(0, testing_amount):
                bsd[j], wsd[j] = self.backpropagation(inputs[j], bs, ws, targets[j], etha)

            for j in range(0, testing_amount):
                for k in range(0, len(nl) - 1):
                    bs[k] = np.subtract(bs[k], bsd[j][k])
                    ws[k] = np.subtract(ws[k], wsd[j][k])

            outputs = [self.calculate_forward(inputs[j], bs, ws) for j in range(0, testing_amount)]
            errors_temp = [self.square_error(outputs[j], targets[j]) for j in range(0, testing_amount)]

            last2_error = last_error
            last_error = error_total
            error_total = 0
            for j in range(0, testing_amount):
                error_total += errors_temp[j]
            error_total /= divide_factor
            errors.append(error_total)
            diffs.append(last_error-error_total)
            diffs2.append((last2_error-last_error)-(last_error-error_total))
        p_x = [i for i in range(0, len(errors))]
        p_y = [float(e) for e in errors]
        p_x_dif = [i for i in range(0, len(diffs))]
        p_y_dif = [d for d in diffs]
        p_x_dif2 = [i for i in range(0, len(diffs2))]
        p_y_dif2 = [d2 for d2 in diffs2]

        if self.show_plot is True:
            plt.figure(0)
            
            ax1 = plt.subplot(111)
            ax1.plot(p_x, p_y)
            plt.yscale('log')
            plt.title('Learning curve of the Neural Network')
            plt.ylabel('Absolute Error from inputs\nto current outputs and targets')
            plt.xlabel('Number of Iterations')
            # plt.figure(1)
            
            plt.figure(1)
            ax2 = plt.subplot(211)
            ax2.plot(p_x_dif, p_y_dif)
            plt.yscale('log')
            # ax2.yscale('log')

            # plt.figure(2)
            ax3 = plt.subplot(212)
            ax3.plot(p_x_dif2, p_y_dif2)
            plt.yscale('log')
            # ax3.yscale('log')
            plt.show()


# ## Testing som functions
# # Create a network Object
# nn = NeuralNetwork()

# # Set some varibales
# nl = [5,4,5]
# bs = nn.get_random_biases_list(nl)
# ws = nn.get_random_weight_list(nl)
# ### Is also possible
# ### bs, ws = nn.get_random_neuron_weight(nl)

# # Get the whole Network
# network = [nl, bs, ws]

# # Set the amount of testing Points (Values)
# testing_amount = 10

# # Get the Inputs and Targets values for learing the Network (here are the values random)
# inputs = [np.array(nn.get_random_matrix(nl[0],1,-3,3,3)) for i in range(0, testing_amount)]
# targets = [np.array(nn.get_random_matrix(nl[-1],1,0.1,0.9,3)) for i in range(0, testing_amount)]

# # Test a loop for learing (is hopfully working ^^)
# print("testing my network with the function improve_network")
# for i in range(0, 4):
#     network = nn.improve_network(network, inputs, targets, 0.05, 1000)
#     print("error of the network is: "+str(nn.get_network_error(network, inputs, targets) / testing_amount / nl[-1]))
# # nn.iterate_with_bgd([5,5,5], 0.05, 10, 1000)
