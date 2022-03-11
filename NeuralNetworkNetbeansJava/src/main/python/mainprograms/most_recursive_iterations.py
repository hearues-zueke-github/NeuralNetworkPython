#! /usr/bin/python2.7

import numpy as np

# print("Hello World!")

m = 10
k = 1
range_a = [0, 5]
range_c = [-5, 5]
range_M = [-5, 5]
a = np.random.randint(range_a[0], range_a[1]+1, (m,))
c = np.random.randint(range_c[0], range_c[1]+1)
M = np.random.randint(range_M[0], range_M[1]+1, (m, k))

print("a = {}".format(a))
print("c = {}".format(c))
print("M = {}".format(M))

def calc_new_value(a, M, c):
    A = [[int(x)**i for i in xrange(1, k+1)] for x in a]
    if A[0][0] > 10**50:
        return "TOOBIG"
    # print("A = {}".format(A))
    MA = M*A
    return np.sum(MA) + c

def calc_new_values(a, M, c, amount):
    a = np.copy(a)
    for _ in xrange(amount):
        a_new = calc_new_value(a[a.shape[0]-m:], M, c)
        if a_new == "TOOBIG":
            print("matrix A is too BIG!!!")
            return -1
        a = np.hstack((a, a_new))
    return a
# Get polynomial Matrix of last m values
a_new = calc_new_values(a, M, c, 30)
print("a_new = {}".format(a_new))
