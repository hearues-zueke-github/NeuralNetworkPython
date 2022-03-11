#! /usr/bin/python2.7

from __init__ import *

def binary_to_int(lbin):
    value = 0
    length = len(lbin)

    for i in xrange(0, length):
        value += lbin[i] * 2**(length - 1 - i)

    return value

def int_to_binlst(value, min_len=0, exactly_values=True):
    """ Returns a binary number as a list of 0 and 1 """
    lbin = []

    if exactly_values:
        if min_len > 0:
            lbin = [0 for _ in xrange(min_len)]
        i = 0
        while value > 0:
            if i < len(lbin):
                lbin[i] = value % 2
            else:
                lbin.append(value % 2)
            value = int(value / 2)
            i += 1
    else:
        d = {0: 0.1, 1: 0.9}
        if min_len > 0:
            lbin = [0.1 for _ in xrange(min_len)]
        i = 0
        while value > 0:
            if i < len(lbin):
                lbin[i] = d[value % 2]
            else:
                lbin.append(d[value % 2])
            value = int(value / 2)
            i += 1

    return lbin[::-1]

def get_inputs_targets_2binadder(bin_digs):
    """ Returns a whole list of inputs and targets of binary numbers """
    max_val = 2**bin_digs

    inputs = [(x, y) for x in xrange(0, max_val) for y in xrange(0, max_val)]
    targets = [x + y for (x, y) in inputs]

    inputs = [int_to_binlst(x, bin_digs)+int_to_binlst(y, bin_digs) for (x, y) in inputs]
    targets = [int_to_binlst(t, bin_digs+1) for t in targets]

    return (inputs, targets)

def get_random_train_test_sets_binadder(bits, exactly_values=True):
    max_val = 2**bits

    inputs = [(x, y) for x in xrange(0, max_val) for y in xrange(0, max_val)]
    targets = [x + y for (x, y) in inputs]

    inputs = np.array([int_to_binlst(x*2**bits+y, 2*bits, True) for (x, y) in inputs])
    targets = np.array([int_to_binlst(t, bits+1, exactly_values) for t in targets])

    max_max_val = max_val**2
    random_indices = np.random.permutation(np.arange(max_max_val))
    split_val = max_max_val / 2

    inputs_train, inputs_test = inputs[random_indices[:split_val]], inputs[random_indices[split_val:]]
    targets_train, targets_test = targets[random_indices[:split_val]], targets[random_indices[split_val:]]

    return inputs_train, targets_train, inputs_test, targets_test

# Change values from a list with the help of a dict
# e.g. l = [1,3,2,4,1,2,1]
#      d = {1: 5, 2: 1}
# so the new list would be: l = [5,3,1,4,5,1,5]
def get_changed_list_vals_dict(l, d):
    return [d[li] if li in d else li for li in l]

# def get_rounded_list(output):
#     l = []
#
#     for o in output:
#         temp = []
#         t = np.transpose(o).tolist()[0]
#         for i in t:
#             temp.append(0 if i < 0.5 else 1)
#         # for
#         l.append(temp)
#     # for
#
#     return l

def check_equivalents(l1, l2):
    correct = 0
    wrong = 0

    for (e1, e2) in zip(l1, l2):
        fail = False
        for (n1, n2) in zip(e1, e2):
            if n1 != n2:
                wrong += 1
                fail = True
                break

        if fail == False:
            correct += 1

    return (correct, wrong)

def get_binaryadder_inputs_targets(digits):
    inp_orig, targ_orig = get_inputs_targets_2binadder(digits)
    d = {0: 0.1, 1: 0.9}

    inp = [l for l in inp_orig]
    targ = [get_changed_list_vals_dict(l, d) for l in targ_orig]

    inp = [np.transpose(np.array([i])) for i in inp]
    targ = [np.transpose(np.array([t])) for t in targ]

    return inp, targ

def get_binaryadder_train_test_sets(digits, exactly_values=True):
    inp_train, targ_train, inp_test, targ_test = get_random_train_test_sets_binadder(digits, exactly_values)
    return inp_train, targ_train, inp_test, targ_test

def calculate_number(neuronal_network, lennum, num1, num2):
    lbin1 = int_to_binlst(num1, lennum)
    lbin2 = int_to_binlst(num2, lennum)

    inp = np.array([lbin1[::-1] + lbin2[::-1]]).transpose()

    out = neuronal_network.calculate_forward(inp, neuronal_network.biases, neuronal_network.weights)

    print("number1 = "+str(num1)+"   in binary = "+"".join([str(i) for i in lbin1]))
    print("number2 = "+str(num2)+"   in binary = "+"".join([str(i) for i in lbin2]))
    
    out = out.transpose().tolist()[0][::-1]
    for o in out:
        if o > 0.5:
            print("1: "+str(100 if o > 0.9 else round((o-0.1)/0.8*100, 1))+"%")
        else:
            print("0: "+str(100 if o > 0.9 else round((0.9-o)/0.8*100, 1))+"%")
    out = [0 if o < 0.5 else 1 for o in out]

    print("result in binary = "+"".join([str(o) for o in out])+"   result is = "+str(binary_to_int(out)))

def get_failure(neuronal_network, targ_orig, iterations = 100):
    nn = neuronal_network
    _, error_list = nn.improve_network_itself(False, nn.inputs, nn.targets, iterations)
    outputs = nn.calculate_forward_many(nn.inputs, nn.biases, nn.weights)
    
    rounded_list = [np.array([0.1 if i < 0.5 else 0.9 for i in o.transpose().tolist()[0]]).transpose() for o in outputs]
    if nn.save_best_network() != 0:
        nn.load_last_network()
        pass
    else:
        nn.show_error_plot(error_list)
        print(str(error_list))

    return check_equivalents(rounded_list, targ_orig)
