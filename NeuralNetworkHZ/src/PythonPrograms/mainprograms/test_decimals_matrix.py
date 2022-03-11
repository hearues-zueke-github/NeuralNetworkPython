#! /usr/bin/python2.7

import numpy as np
from random import randint
import decimal
from decimal import Decimal as dec

def nparray_dec_to_float(array_dec):
    list_float = []
    for ad in array_dec:
        list_dec = ad.tolist()
        list_float.append([float(str(i)) for i in list_dec])
    # for
    return np.array(list_float)
# def dec_to_float

def nparray_float_to_dec(array_float):
    list_dec = []
    for af in array_float:
        list_float = af.tolist()
        list_dec.append([dec(str(i)) for i in list_float])
    # for
    return np.array(list_dec)
# def dec_to_float

decimal.getcontext().prec = 50
np.set_printoptions(precision=30)

a = np.array([[1,2,3,4,dec("5.000000000000000000000000000000000001")]])
print("a = "+str(a))

b = np.array([[2,3,4,5,dec("8.")]])
print("a = "+str(b))

c = a * b
# c = np.add(a, b)
print("c = "+str(c))

d = np.array([[1,2,3],[5,6,4],[9,8,7]])
