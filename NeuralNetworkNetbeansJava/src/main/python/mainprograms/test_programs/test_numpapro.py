#! /home/haris/anaconda/bin/python2.7
#! /usr/bin/python2.7

import numbapro
from numbapro import jit, int32, float32, complex64

@jit(complex64(int32, float32, complex64), target="cpu")
def bar(a, b, c):
   return a + b  * c

@jit(complex64(int32, float32, complex64)) # target kwarg defaults to "cpu"
def foo(a, b, c):
   return a + b  * c

print(numbapro)
print(numbapro.numba)
print(numbapro.numba.cuda)
print(numbapro.numba.cuda.get_current_device())
print(foo)
print(foo(1, 2.0, 3.0j))

# import math
# from numbapro import vectorize, cuda
# import numpy as np

# @vectorize(['float32(float32, float32, float32)',
#             'float64(float64, float64, float64)'],
#            target='gpu')
# def cu_discriminant(a, b, c):
#     return math.sqrt(b ** 2 - 4 * a * c)

# N = 1e+4
# dtype = np.float32

# # prepare the input
# A = np.array(np.random.sample(N), dtype=dtype)
# B = np.array(np.random.sample(N) + 10, dtype=dtype)
# C = np.array(np.random.sample(N), dtype=dtype)

# D = cu_discriminant(A, B, C)

# print(D)  # print result
