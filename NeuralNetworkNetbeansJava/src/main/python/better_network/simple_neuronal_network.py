#! /usr/bin/python2.7

import sys
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy

f = lambda x: (1 / (1 + np.exp(-x)))
fi = lambda x: f(x) * (1 - f(x))

vec_f = np.vectorize(f)
vec_fi = np.vectorize(fi)

def get_quadratic_error(inputs, weights, targets):
    outputs = calc_forward_many(inputs, weights)

    return sum(map(lambda (o, t): ((o-t)**2).sum(), zip(outputs, targets)))

def calc_forward_many(inputs, weights): #biases, weights):
    ""
    # wit = np.array(biases.tolist()+weights.tolist()).transpose()
    weights = weights.transpose()
    targets = [vec_f(np.dot(weights, x)) for x in inputs]

    return targets

def calc_new_weights(inputs, biases, weights, targets, learn_rate):
    new_weights = deepcopy(np.array(biases.tolist()+weights.tolist()))

    activations = [np.dot(new_weights.transpose(), inp) for inp in inputs]
    outputs = calc_forward_many(inputs, new_weights)

    new_weights = new_weights.transpose()

    for inp, act, out, targ in zip(inputs, activations, outputs, targets):
        new_weights = new_weights - learn_rate * np.dot((out - targ)*vec_fi(act), inp.transpose())

    return new_weights.transpose()

def calc_new_weights_iterations(inputs, weights, targets, learn_rate, iterations):
    new_weights = deepcopy(weights)
    # print("new weights =\n"+str(new_weights))
    error_list = []
    for _ in xrange(0, iterations):
        activations = [np.dot(new_weights.transpose(), inp) for inp in inputs]
        # outputs = calc_forward_many(inputs, np.array(new_weights.tolist()[0]), np.array(new_weights.tolist()[1:]))
        # outputs = calc_forward_many(inputs, biases, weights)
        outputs = calc_forward_many(inputs, new_weights)

        # new_weights = new_weights.transpose()

        for inp, act, out, targ in zip(inputs, activations, outputs, targets):
            new_weights = new_weights - learn_rate * np.dot((out - targ)*vec_fi(act), inp.transpose()).transpose()

        error_list.append(get_quadratic_error(inputs, new_weights, targets))
    return new_weights, error_list

def main(argv):
    ""
    inp = 5
    out = 3

    b = np.random.uniform(-3.0, 3.0, (1, out))
    w = np.random.uniform(-3.0, 3.0, (inp, out))

    wi = np.array(b.tolist()+w.tolist())

    inputs = [np.array([[1]]+np.random.random((inp, 1)).tolist()) for _ in xrange(0, 4)]

    # print("b =\n"+str(b))
    # print("w =\n"+str(w))
    print("wi =\n"+str(wi))
    # print("inputs =\n"+str(inputs))

    targets = calc_forward_many(inputs, wi)

    # print("targets =\n"+str(targets))

    bn = np.random.uniform(-3.0, 3.0, (1, out))
    wn = np.random.uniform(-3.0, 3.0, (inp, out))

    # print("bn =\n"+str(bn))
    # print("wn =\n"+str(wn))
    wni = np.array(bn.tolist() + wn.tolist())

    plt.yscale("log")
    print("wni =\n"+str(wni))
    for _ in xrange(0, 10):
        wni, error_list = calc_new_weights_iterations(inputs, wni, targets, 0.1, 300)
        print("wni =\n"+str(wni))
        print("error_list = \n"+str(error_list))
        plt.plot(error_list)
        plt.show(block=False)
    raw_input()
# def main

if __name__ == "__main__":
    main(sys.argv)
# if
