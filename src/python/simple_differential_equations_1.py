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
	# A and x are matrices.
	# A = R**(rows x cols)
	# x = R**(cols x cols2)
	rows = 5
	cols = 3
	cols2 = 2

	# Following f and cost function are given: f = cos(A.dot(x)); cost(x) = 1/2 * sqrt(sum(f(x)**2))**2 # also know as euclidian distance squared
	# idea: minimize x s.t. cost(x) min is!

	# A = (np.arange(0, rows*cols).reshape((rows, cols)) + 1).astype(np.float64)
	A = np.random.uniform(-1, 1, (rows, cols))
	# x = (np.arange(0, cols).reshape((cols, )) + 1).astype(np.float64)
	x = np.random.uniform(-1, 1, (cols, 2))

	print(f"A:\n{A}")
	print(f"x:\n{x}")

	def g_func(x):
		return np.sin(x)

	def g_derv_func(x):
		return np.cos(x)

	def f_func(A, x):
		return g_func(A.dot(x))

	def f_derv_func(A, x):
		v_A_x = A.dot(x)
		return g_func(v_A_x) * g_derv_func(v_A_x)

	v_A_x = f_func(A, x)
	print(f"v_A_x:\n{v_A_x}")

	def f_cost(v):
		return 1. / 2 * np.sum(v**2)

	cost_v = f_cost(v_A_x)
	print(f"cost_v:\n{cost_v}")

	def f_nabla(A, x):
		return A.transpose().dot(f_derv_func(A, x))
		# return A.transpose().dot(f_derv_func(A, x))

	nabla_x = f_nabla(A, x)

	print(f"nabla_x:\n{nabla_x}")

	def numerical_gradient(A, x_orig, epsilon=0.000001):
		x = x_orig.copy()
		nabla_x_num = x.copy()

		for i in range(0, x.shape[0]):
			for j in range(0, x.shape[1]):
				x[i, j] += epsilon
				cost_1 = f_cost(f_func(A, x))
				x[i, j] -= epsilon * 2
				cost_2 = f_cost(f_func(A, x))
				x[i, j] += epsilon

				differential = (cost_1 - cost_2) / (2. * epsilon)
				nabla_x_num[i, j] = differential

		return nabla_x_num

	# nabla_x_num_1 = numerical_gradient(A, x, epsilon=0.001)
	# print(f"epsilon: 0.001, nabla_x:\n{nabla_x_num_1}")

	nabla_x_num_2 = numerical_gradient(A, x, epsilon=0.0001)
	print(f"\nepsilon: 0.0001, nabla_x:\n{nabla_x_num_2}")

	# nabla_x_num_3 = numerical_gradient(A, x, epsilon=0.00001)
	# print(f"epsilon: 0.00001, nabla_x:\n{nabla_x_num_3}")
