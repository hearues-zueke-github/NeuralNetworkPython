#! /usr/bin/python2.7

import numpy as np

from time import time

nl = [10000, 800, 600, 500, 400, 10]

X = np.random.random(nl[:2])

Wbs = [np.random.random((nl[i] + 1, nl[i + 1])) for i in xrange(1, len(nl) - 1)]
Ws = [np.random.random((nl[i], nl[i + 1])) for i in xrange(1, len(nl) - 1)]
bs = [np.random.random((nl[i + 1], )).reshape((1, -1)) for i in xrange(1, len(nl) - 1)]

f_sig = lambda X: 1. / (1 + np.exp(-X))

print("X.shape: {}".format(X.shape))

for i, W in enumerate(Wbs):
    print("i: {}, W.shape: {}".format(i, W.shape))

start_time = time()
Y1 = X
for W, b in zip(Ws, bs):
    # for W in Wbs:
    A = Y1.dot(W)+b
    # A = np.hstack((np.ones((Y.shape[0], 1)), Y)).dot(W)
    Y1 = f_sig(A)
end_time = time()

print("Y1.shape: {}".format(Y1.shape))
print("end_time-start_time: {:2.5}s".format(end_time-start_time))

start_time = time()
Y2 = X
for W, b in zip(Ws, bs):
    # for W in Wbs:
    A = Y2.dot(W)+b
    # A = np.hstack((np.ones((Y.shape[0], 1)), Y)).dot(W)
    Y2 = f_sig(A)
end_time = time()

print("Y2.shape: {}".format(Y2.shape))
print("end_time-start_time: {:2.5}s".format(end_time-start_time))
