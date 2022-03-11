#! /usr/bin/python2.7

import numpy as np

import UtilsBinary

def print_2d_array(array):
    str_text = "("

    for row in array[:-1]:
        str_text += str(row)+"\n "
    str_text += str(array[-1])+")"

    print("{}".format(str_text))

def get_specific_function_binary(ls):
    def get_x_binary(l):
        expr = ""
        for i, j in enumerate(l):
            if j > 0 and np.random.randint(0, 2) == 0:
                if expr != "":
                    expr += "&"
                if j == 1:
                    expr += "~"
                    # expr += "~(x["+str(i)+"].astype(np.bool))"
                # else:
                expr += "x["+str(i)+"]"

        return expr

    str_expr = "lambda x: ("

    while True:
        mult_exprs = [get_x_binary(l) for l in ls]

        mult_exprs = [mult_expr for mult_expr in mult_exprs if len(mult_expr) != 0]

        if len(mult_exprs) != 0:
            break

    str_expr += mult_exprs[0]
    for mult_expr in mult_exprs[1:]:
        str_expr += "|"+mult_expr

    str_expr += ")"
    # str_expr += ").astype(np.int)"

    return str_expr, eval(str_expr)

def get_a_unique(args_amount):
    zero_tuple = (0, )*args_amount
    # max_len = args_amount
    max_len = 2**(int(args_amount/1.5))
    if max_len < 2:
        max_len = 2
    print("max_len: {}".format(max_len))
    get_rnd_len = lambda: np.random.randint(2, max_len+1)
    while True:
        a = np.random.randint(0, 3, (get_rnd_len(), args_amount))
        a_sorted = sorted(a.tolist())
        
        a_unique = sorted(list(set(list(map(tuple, a)))))
        if zero_tuple == a_unique[0]:
            a_unique.pop(0)
        
        if len(a_unique) > 0:
            break

    return a_unique

def get_binary_logical_functions_X_T(input_neurons, output_neurons, with_random=False, with_permutation=False, random_max_range=0.2):
    a_uniques = []
    a_uniques_set = []

    str_exprs = []
    fs = []

    # input_neurons = 8
    # output_neurons = 3
    for i in xrange(0, output_neurons):
        a_unique_new = get_a_unique(input_neurons)
        a_unique_set_new = set(a_unique_new)
        if len(a_uniques) > 0:
            while a_unique_set_new in a_uniques_set:
                a_unique_new = get_a_unique(input_neurons)
                a_unique_set_new = set(a_unique_new)
            
        a_uniques.append(a_unique_new)
        a_uniques_set.append(a_unique_set_new)

        str_expr, f = get_specific_function_binary(a_unique_new)
        str_exprs.append(str_expr)
        fs.append(f)

        print("i: {}".format(i))
        print("  a_unique_new:")
        # print_2d_array(a_unique_new)
        print("  str_expr: {}".format(str_expr))
        # print("  f: {}\n".format(f))

    X = UtilsBinary.get_all_combinations(input_neurons).T.astype(np.bool)

    T = fs[0](X)
    for f in fs[1:]:
        T = np.vstack((T, f(X)))

    X = X.T.astype(np.int)
    T = T.T.astype(np.int)

    if with_random:
        def get_new_X(X):
            X_random = np.random.random(X.shape)*random_max_range
            X_random[X==1] = 1 - X_random[X==1]
            return X_random

        X_new = X.copy()
        T_new = T.copy()
        for _ in xrange(0, 3):
            X_new = np.vstack((X_new, get_new_X(X)))
            T_new = np.vstack((T_new, T))
        X = X_new
        T = T_new

    if with_permutation:
        idxs = np.random.permutation(np.arange(0, X.shape[0]))
        X = X[idxs]
        T = T[idxs]

    print("X:\n{}".format(X))
    print("T:\n{}".format(T))

    return X, T
