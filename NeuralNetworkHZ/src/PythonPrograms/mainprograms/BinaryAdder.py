#! /usr/bin/python2.7

import numpy as np

def binary_incrementer(lbin):
    rest = 1
    lbininc = []
    for b in lbin:
        if b + rest == 2:
            lbininc.append(0)
            rest = 1
        elif b + rest == 1:
            lbininc.append(1)
            rest = 0
        else:
            lbininc.append(0)
            rest = 0
        # if
    # for

    if rest == 1:
        lbininc.append(1)
    # if

    return lbininc
# def binary_incrementer

def binary_adder(lbin1, lbin2):
    len1 = len(lbin1)
    len2 = len(lbin2)
    lbin3 = []

    if len1 < len2:
        lbin1l = lbin1
        lbin1r = []
        lbin2l = lbin2[:len1]
        lbin2r = lbin2[len1:]
    elif len1 > len2:
        lbin1l = lbin1[:len2]
        lbin1r = lbin1[len2:]
        lbin2l = lbin2
        lbin2r = []
    else:
        lbin1l = lbin1
        lbin1r = []
        lbin2l = lbin2
        lbin2r = []
    # if

    rest = 0
    for (b1, b2) in zip(lbin1l, lbin2l):
        if b1 + b2 + rest == 3:
            lbin3.append(1)
            rest = 1
        elif b1 + b2 + rest == 2:
            lbin3.append(0)
            rest = 1
        elif b1 + b2 + rest == 1:
            lbin3.append(1)
            rest = 0
        else:
            lbin3.append(0)
            rest = 0
        # if
    # for

    lbinrest = []
    if len1 < len2:
        lbinrest = lbin2r
    elif len1 > len2:
        lbinrest = lbin1r
    # if

    for b in lbinrest:
        if b + rest == 2:
            lbin3.append(0)
            rest = 1
        elif b + rest == 1:
            lbin3.append(1)
            rest = 0
        else:
            lbin3.append(0)
            rest = 0
        # if
    # for

    if rest == 1:
        lbin3.append(1)
    # if

    return lbin3
# def binary_adder

def binary_to_int(lbin):
    value = 0
    length = len(lbin)

    for i in xrange(0, length):
        value += lbin[i] * 2**i
    # for

    return value
# def binary_to_int

def int_to_binlst(value, min_len = 0):
    lbin = []
    if min_len > 0:
        lbin = [0 for _ in xrange(min_len)]
    # if
    
    i = 0
    while value > 0:
        if i < len(lbin):
            lbin[i] = value % 2
        else:
            lbin.append(value % 2)
        # if
        value = int(value / 2)
        i += 1
    # while

    return lbin
# def int_to_binary

def int_to_binstr(value):
    sbin = ""
    
    while value > 0:
        sbin += str(value % 2)
        value = int(value / 2)
    # while

    return sbin
# def int_to_binary

def get_inputs_targets_2binadder(bin_digs):
    max_val = 2**bin_digs
    inputs = []
    for y in xrange(0, max_val):
        for x in xrange(0, max_val):
            inputs.append((x,y))
        # for
    # for
    # inputs = [(x, y) for (x,y) in zip(xrange(0, max_val), xrange(0, max_val))]
    targets = [x + y for (x,y) in inputs]

    inputs = [(int_to_binlst(x,bin_digs)+int_to_binlst(y,bin_digs))[::-1] for (x,y) in inputs]
    targets = [int_to_binlst(t,bin_digs+1)[::-1] for t in targets]

    # inputs = [np.array([inp]) for inp in inputs]
    # targets = [np.array([targ]) for targ in targets]

    return (inputs, targets)
# def create_inputs_targets_2binadder

# Change values from a list with the help of a dict
# e.g. l = [1,3,2,4,1,2,1]
#      d = {1: 5, 2: 1}
# so the new list would be: l = [5,3,1,4,5,1,5]
def get_changed_list_vals_dict(l, d):
    return [d[li] if li in d else li for li in l]
# def get_changed_list_vals_dict

def get_rounded_list(output):
    l = []

    for o in output:
        temp = []
        t = np.transpose(o).tolist()[0]
        for i in t:
            temp.append(0 if i < 0.5 else 1)
        # for
        l.append(temp)
    # for

    return l
# def get_rounded_list

def check_equivalents(l1, l2):
    correct = 0
    wrong = 0

    for (e1, e2) in zip(l1, l2):
        if e1 == e2:
            correct +=1
        else:
            wrong += 1
        # if
    # for

    return (correct, wrong)
# def check_equivalents

def get_binaryadder_inputs_targets(digits):
    inp_orig, targ_orig = get_inputs_targets_2binadder(digits)
    d = {0: 0.1, 1: 0.9}

    # inp = [get_changed_list_vals_dict(l, d) for l in inp_orig]
    inp = [l for l in inp_orig]
    targ = [get_changed_list_vals_dict(l, d) for l in targ_orig]

    inp = [np.transpose(np.array([i])) for i in inp]
    targ = [np.transpose(np.array([t])) for t in targ]

    return inp, targ
# def get_binaryadder_inputs_targets

def get_failure(neuronal_network, targ_orig, iterations = 100):
    nn = neuronal_network
    nn.improve_network_itself(False, nn.inputs, nn.targets, iterations)
    outputs = nn.calculate_forward_many(nn.inputs, nn.biases, nn.weights)
    rounded_list = get_rounded_list(outputs)
    if nn.save_best_network() != 0:
        nn.load_last_network()
    else:
        nn.show_error_plot()
    # if
    return check_equivalents(rounded_list, targ_orig)
# def get_failure
