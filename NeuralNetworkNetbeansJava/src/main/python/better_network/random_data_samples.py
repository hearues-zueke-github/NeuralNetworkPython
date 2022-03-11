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

nn = NeuralNetwork()

inputs = 4
outputs = 3

nl = [inputs, 100, 100, outputs]

X = np.random.random((2000, 4))*2-1

T = np.vstack((X[:, 0]+X[:, 1],
               # X[:, 2]+X[:, 3],
               # X[:, 1]+X[:, 2]-X[:, 3],
               X[:, 2]+X[:, 3],
               X[:, 0]+X[:, 3])).T

T_sig = nn.f_sig(T)

# print("X:\n{}".format(X))
# print("T:\n{}".format(T))
# print("T_sig:\n{}".format(T_sig))

# T_bools = np.zeros(T_sig.shape)
# T_bools[T_sig > 0.5] = 1
# print("T_bools:\n{}".format(T_bools))

# split in train, valid, test

idx_1 = int(0.6*X.shape[0])
idx_2 = int(0.8*X.shape[0])

print("idx_1: {}".format(idx_1))
print("idx_2: {}".format(idx_2))

X_train, T_train = X[:idx_1], T[:idx_1]
X_valid, T_valid = X[idx_1:idx_2], T[idx_1:idx_2]
X_test, T_test = X[idx_2:], T[idx_2:]

nn.init_bws(X_train, T_train, nl[1:-1])
nn.calc_cost = nn.f_cecf
# nn.calc_missclass = nn.f_missclass
nn.calc_missclass = nn.f_missclass_regression
# nn.calc_missclass = nn.f_missclass_onehot

nn.fit_network(X_train, T_train, X_test, T_test)
