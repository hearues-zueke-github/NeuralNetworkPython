#! /usr/bin/python3.10.2

# pip installed libraries
import dill
import gzip
import os
import requests
import string
import subprocess
import sys
import time
import yaml

import multiprocessing as mp
import numpy as np
import pandas as pd

from io import StringIO
from memory_tempfile import MemoryTempfile

from numpy.random import Generator, PCG64
from scipy.sparse import csr_matrix, coo_matrix

from PIL import Image
from typing import Dict

HOME_DIR = os.path.expanduser("~")
PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_PROJECT_PATH_DIR = os.path.join(PATH_ROOT_DIR, '../..')
TEMP_DIR = MemoryTempfile().gettempdir()

if __name__ == '__main__':
	# a very small neural network given
	nl = [3, 5, 4, 2]
	l_bw = [np.random.uniform(-1, 1, (l2, l1 + 1)) for l1, l2 in zip(nl[:-1], nl[1:])]

	# sigmoid function
	def f_sig(z):
		return 1. / (1 + np.exp(-z))
	def f_sig_derv(z):
		return f_sig(z) * (1 - f_sig(z))

	# as a hidden function the tanh is used as an example
	def f_hidden(x):
		return np.tanh(x)
	def f_hidden_derv(x):
		return 1 - np.tanh(x)**2

	print(f"l_bw: {l_bw}")

	# we have only one x value given, therefore x is a vector of length nl[0] = 3
	x = np.random.uniform(-1, 1, (nl[0]))
	# we have only one t value given, therefore t is a vector of length nl[-1] = 2
	t = np.random.randint(0, 2, (nl[-1], )).astype(np.float64)

	def f_forward(x):
		a = x
		for bw in l_bw[:-1]:
			a_1 = np.hstack(((1, ), a))
			z = bw.dot(a_1)
			a = f_hidden(z)

		a_1 = np.hstack(((1, ), a))
		z = l_bw[-1].dot(a_1)
		a = f_sig(z)

		return a

	y = f_forward(x)
	print(f"x: {x}")
	print(f"y: {y}")

	def f_cost(y, t):
		return np.sum(np.nan_to_num(-t*np.log(y)-(1-t)*np.log(1-y)))

	def f_delta(z, y, t):
		return (y - t)

	def backprop(x, t):
		y = x
		l_y = [x]
		l_z = []
		for bw in l_bw[:-1]:
			y = np.hstack(((1, ), y))
			z = np.dot(bw, y)
			l_z.append(z)
			y = f_hidden(z)
			l_y.append(y)

		y = np.hstack(((1, ), y))
		z = np.dot(l_bw[-1], y)
		l_z.append(z)
		y = f_sig(z)
		l_y.append(y)

		delta = f_delta(l_z[-1], l_y[-1], t)
		l_nabla_bw = [np.zeros(bw.shape) for bw in l_bw]
		l_nabla_bw[-1] = np.dot(delta.reshape((-1, 1)), np.hstack(((1, ), l_y[-1-1].transpose())).reshape((1, -1)))

		for l in range(2, len(nl)):
			z = l_z[-l]
			sp = f_hidden_derv(z)
			delta = np.dot(l_bw[-l+1][:, 1:].transpose(), delta) * sp
			l_nabla_bw[-l] = np.dot(delta.reshape((-1, 1)), np.hstack(((1, ), l_y[-l-1].transpose())).reshape((1, -1)))
		return l_nabla_bw

	l_nabla_bw = backprop(x, t)
	print(f"l_nabla_bw: {l_nabla_bw}")

	def numerical_gradient(x, epsilon=0.000001):
		l_nabla_bw_numeric = [np.zeros(bw.shape) for bw in l_bw]
		for nabla_bw_numeric, bw in zip(l_nabla_bw_numeric, l_bw):
			for i in range(0, bw.shape[0]):
				for j in range(0, bw.shape[1]):
					bw[i, j] += epsilon
					cost_1 = f_cost(f_forward(x), t)
					bw[i, j] -= epsilon * 2
					cost_2 = f_cost(f_forward(x), t)
					bw[i, j] += epsilon

					differential = (cost_1 - cost_2) / (2. * epsilon)
					nabla_bw_numeric[i, j] = differential

		return l_nabla_bw_numeric

	l_nabla_bw_numeric = numerical_gradient(x, epsilon=0.0001) # epsilon 10**-4 is good for a numerical approximation!
	print(f"l_nabla_bw_numeric: {l_nabla_bw_numeric}")

	# if both nablas are really close, only then we can say, that the backprop algo is calculating the close correct values!
	l_nabla_diff = [nabla_bw - nabla_bw_numeric for nabla_bw, nabla_bw_numeric in zip(l_nabla_bw, l_nabla_bw_numeric)]
	print(f"l_nabla_diff: {l_nabla_diff}")
