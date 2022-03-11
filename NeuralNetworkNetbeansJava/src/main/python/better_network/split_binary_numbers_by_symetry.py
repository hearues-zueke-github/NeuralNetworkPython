#! /usr/bin/python2.7

import csv
import gzip
import os
import select
import sys
import time

import numpy as np
import multiprocessing as mp
import pickle as pkl

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr

from copy import deepcopy
from PIL import Image
from UtilsBinary import *

bits = 3
X, T = get_binary_adder_numbers(bits)

print("X:\n{}".format(X))
print("T:\n{}".format(T))

X_sums = np.sum(X*2**np.arange(2*bits-1, -1, -1), axis=1)
print("X_sums:\n{}".format(X_sums))

X_switched = np.hstack((X[:, bits:2*bits], X[:, 0:bits]))
print("X_switched:\n{}".format(X_switched))
X_switched_sums = np.sum(X_switched*2**np.arange(2*bits-1, -1, -1), axis=1)
print("X_switched_sums:\n{}".format(X_switched_sums))
X_switched_sums_ids = np.hstack((np.arange(0, 2**(2*bits)).reshape((-1, 1)), X_switched_sums.reshape((-1, 1))))
print("X_switched_sums_ids:\n{}".format(X_switched_sums_ids))

X_both_equal = np.sum(X[:, bits:2*bits]==X[:, 0:bits], axis=1)==bits
print("X_both_equal:\n{}".format(X_both_equal))
