#! /usr/bin/python3

import time

import numpy as np

# from ... import Utils as utils

a = np.random.rand(500, 10000)
b = np.random.rand(10000, 3000)

start = time.time()
c = np.dot(a, b)
end = time.time()

print("Needed time for execution: "+str(end - start)+"s")
