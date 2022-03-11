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

def get_function_combination_string(X, m):
    n = X.shape[1]

    def get_random_column_idx():
        if n > 4:
            columns_idx = np.sort(np.random.permutation(np.arange(0, n))[:np.random.randint(1, 4)])
        else:
            columns_idx = np.sort(np.random.permutation(np.arange(0, n))[:np.random.randint(1, n)])

        return columns_idx

    def get_combined_column_idxs():
        amount_or = np.random.randint(1, 5)

        t = []
        for _ in xrange(0, amount_or):
            t.append(get_random_column_idx())

        return t

    def get_function_string(t):
        string = "("
        string += "{}".format(t[0][0])
        for i in t[0][1:]:
            string += " & {}".format(i)
        string += ")"

        for l in t[1:]:
            string += " | ("
            string += "{}".format(l[0])
            for i in l[1:]:
                string += " & {}".format(i)
            string += ")"

        return string

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

    t = get_combined_column_idxs()
    expr = get_eval_expr_string(t)
    T = eval(expr)
    for i in xrange(0, m):
        t_ = get_combined_column_idxs()
        expr_ = get_eval_expr_string(t_)
        T_ = eval(expr_)
        
        T = np.vstack((T, T_))

    return T.T

def get_random_binary_functions(X, m):
    n = X.shape[1]

    def get_one_and_function(X):
        if n > 3:
            columns_idx = np.random.permutation(np.arange(0, n))[:np.random.randint(1, 3)]
        else:
            columns_idx = np.random.permutation(np.arange(0, n))[:np.random.randint(1, n)]

        expr = "X[:, {}]".format(columns_idx[0])
        for i in columns_idx[1:]:
            expr += "&X[:, {}]".format(i)

        print("expr: {}".format(expr))
        return eval(expr)

    def get_combined_and_or_function(X):
        amount_or = np.random.randint(1, 4)

        t = get_one_and_function(X)
        for _ in xrange(1, amount_or):
            t |= get_one_and_function(X)

        return t

    print("X:\n{}".format(X))

    T = get_combined_and_or_function(X)
    print("T:\n{}".format(T))
    for _ in xrange(1, m):
        T_ = get_combined_and_or_function(X)
        print("T_:\n{}".format(T_))

        T = np.vstack((T, T_))

    T = T.T
    print("T:\n{}".format(T))

    return T

nn = NeuralNetwork()

inputs = 5
outputs = 5

nl = [inputs, 100, 100, outputs]

# X = np.random.randint(0, 2, (800, inputs))
# T = get_function_combination_string(X, outputs)

# def get_noisy_X(X, factor=0.1):
#     X1 = np.zeros(X.shape)
#     noise_0 = np.random.random(X.shape)*factor
#     noise_1 = np.random.random(X.shape)*factor+1.-factor
#     X1[X == 0] = noise_0[X == 0]
#     X1[X == 1] = noise_1[X == 1]

#     return X1

# X1 = get_noisy_X(X, factor=0.1)
# X2 = get_noisy_X(X, factor=0.2)
# X3 = get_noisy_X(X, factor=0.3)

# # X2 = np.zeros(X.shape)
# # X2[X == 0] = 0.1
# # X2[X == 1] = 0.9

# X = np.vstack((X, X1, X2, X3))
# T = np.vstack((T, T, T, T))

# shuffle_idx = np.random.permutation(X.shape[0])
# X = X[shuffle_idx]
# T = T[shuffle_idx]

# # print("X:\n{}".format(X))
# # print("T:\n{}".format(T))

# # raw_input()

# idx_1 = int(0.6*X.shape[0])
# idx_2 = int(0.8*X.shape[0])

# print("idx_1: {}".format(idx_1))
# print("idx_2: {}".format(idx_2))

# X_train, T_train = X[:idx_1], T[:idx_1]
# X_valid, T_valid = X[idx_1:idx_2], T[idx_1:idx_2]
# X_test, T_test = X[idx_2:], T[idx_2:]

X_train, T_train, X_valid, T_valid, X_test, T_test = get_X_T_binary_regression_data(inputs, 100, outputs)

# print("X_train:\n{}".format(X_train))
# print("T_train:\n{}\n".format(T_train))
# print("X_valid:\n{}".format(X_valid))
# print("T_valid:\n{}\n".format(T_valid))
# print("X_test:\n{}".format(X_test))
# print("T_test:\n{}\n".format(T_test))

# raw_input()

nn.init_bws(X_train, T_train, nl[1:-1])
nn.calc_cost = nn.f_cecf
nn.calc_missclass = nn.f_missclass

nn.fit_network_basic(X_train, T_train, X_test, T_test)
# nn.fit_network(X_train, T_train, X_test, T_test)
