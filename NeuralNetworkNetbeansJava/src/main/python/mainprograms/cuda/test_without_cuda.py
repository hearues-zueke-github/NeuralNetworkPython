#! /usr/bin/python

import math

import numpy as np

from datetime import datetime
from random import randint, uniform
from numba import vectorize, float64

@vectorize([float64(float64, float64)])
def f(x, y):
    return x + y

def test_run(a, b, x=4, y=6):
    print("a = "+str(a))
    print("b = "+str(b))
    print("x = "+str(x))
    print("y = "+str(y))
    for _ in xrange(0, 100000): x = 0
# def

@vectorize([float64(float64, float64)])
def test_array_vectorize(a, b):
    c = a + b * 2
    return math.sqrt(c)
# def test_array_vectorize

def test_array_forloop(a, b):
    new = []
    
    for x, y in zip(a, b):
        c = x + y * 2
        new.append(math.sqrt(c))

    return np.array(new)

def test_time(f, *a, **k):
    start = datetime.now()
    f(*a, **k)
    end = datetime.now()
    return end - start
# def test_time

# start = datetime.now()
# for _ in xrange(0, 100000): x = 0
# end = datetime.now()

# print("test function takes: "+str(test_time(test_run, *[4,5], **{"y": 7, "x": 12})))
def get_random_list(length):
    return [np.float64(uniform(0.0, 5.0)) for _ in xrange(0, length)]

def compare_loop_vectorize():
    a = get_random_list(10**7)
    b = get_random_list(10**7)

    time1 = test_time(test_array_forloop, *[a, b], **{})
    time2 = test_time(test_array_forloop, *[a, b], **{})

    print("time 1: loop: "+str(time1))
    print("time 2: vect: "+str(time2))
# def compare_loop_vectorize

compare_loop_vectorize()
