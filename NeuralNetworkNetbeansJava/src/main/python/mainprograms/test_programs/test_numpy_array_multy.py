#! /usr/bin/python3

import numpy as np

a = np.random.rand(2000, 10000)
b = np.random.rand(10000, 3000)

c = np.dot(a, b)
