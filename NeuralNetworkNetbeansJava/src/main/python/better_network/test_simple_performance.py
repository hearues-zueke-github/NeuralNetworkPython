#! /usr/bin/python2.7

import numpy as np

from time import time

start_time = time()

print("Create A and B matrix")
n = 500
B = np.random.random((n, n))
A = np.random.random((n, n))

iterations = 20
print("Iterate {} times".format(iterations))
for _ in xrange(0, iterations):
    A = A.dot(B)
    B = B.dot(A)

end_time = time()

print("Taken time: {:2.5}s".format(end_time-start_time))
