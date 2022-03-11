#! /usr/bin/python2.7
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath('../utils'))
import HashFunction as hfunc
import Utils
import UtilsBinary

from neuralnetwork import NeuralNetwork

def get_X_T_binary_regression_data(inputs, n, outputs):
    X_train = np.random.randint(0, 2, (int(n*0.6), inputs))
    X_valid = np.random.randint(0, 2, (int(n*0.2), inputs))
    X_test = np.random.randint(0, 2, (int(n*0.2), inputs))

    def get_eval_expr_string(t):
        string = "("
        string += "X[:, {}]".format(t[0][0])
        for i in t[0][1:]:
            string += " & X[:, {}]".format(i)
        string += ")"

        for l in t[1:]:
            string += " | ("
            string += "X[:, {}]".format(l[0])
            for i in l[1:]:
                string += " & X[:, {}]".format(i)
            string += ")"

        return string

    def get_combined_column_idxs():
        amount_or = np.random.randint(1, 5)

        def get_random_column_idx():
            if inputs > 4:
                columns_idx = np.sort(np.random.permutation(np.arange(0, inputs))[:np.random.randint(1, 4)])
            else:
                columns_idx = np.sort(np.random.permutation(np.arange(0, inputs))[:np.random.randint(1, inputs)])

            return columns_idx

        t = []
        for _ in xrange(0, amount_or):
            t.append(get_random_column_idx())

        return t

    def get_T(X, ts):
        expr = get_eval_expr_string(ts[0])
        T = eval(expr)
        for i in xrange(1, outputs):
            expr_ = get_eval_expr_string(ts[i])
            T_ = eval(expr_)
            
            T = np.vstack((T, T_))

        return T.T

    def get_noisy_X(X, factor=0.1):
        X1 = np.zeros(X.shape)
        noise_0 = np.random.random(X.shape)*factor
        noise_1 = np.random.random(X.shape)*factor+1.-factor
        X1[X == 0] = noise_0[X == 0]
        X1[X == 1] = noise_1[X == 1]

        return X1

    def extend_X_T_with_noise(X, T):
        X1 = get_noisy_X(X, factor=0.1)
        X2 = get_noisy_X(X, factor=0.2)
        X3 = get_noisy_X(X, factor=0.3)
        X = np.vstack((X, X1, X2, X3))
        T = np.vstack((T, T, T, T))

        shuffle_idx = np.random.permutation(X.shape[0])
        X = X[shuffle_idx]
        T = T[shuffle_idx]

        return X, T

    ts = [get_combined_column_idxs() for _ in xrange(0, outputs)]

    T_train = get_T(X_train, ts)
    T_valid = get_T(X_valid, ts)
    T_test = get_T(X_test, ts)

    X_train, T_train = extend_X_T_with_noise(X_train, T_train)
    X_valid, T_valid = extend_X_T_with_noise(X_valid, T_valid)
    X_test, T_test = extend_X_T_with_noise(X_test, T_test)

    return X_train, T_train, X_valid, T_valid, X_test, T_test

nn = NeuralNetwork()

inputs = 8
outputs = 3

nl = [inputs, 300, 300, outputs]

X_train, T_train, X_valid, T_valid, X_test, T_test = get_X_T_binary_regression_data(inputs, 800, outputs)

nn.init_bws(X_train, T_train, nl[1:-1])
nn.calc_cost = nn.f_cecf
nn.calc_missclass = nn.f_missclass

nn.fit_network_basic(X_train, T_train, X_test, T_test)
# nn.fit_network(X_train, T_train, X_test, T_test)
