#! /usr/bin/python2.7

import os
import select
import sys

import numpy as np
import multiprocessing as mp

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

from copy import deepcopy
from PIL import Image

sig = lambda X: 1. / (1 + np.exp(-X))
sig_dev = lambda X: sig(X) * (1 - sig(X))
calc_rmse = lambda Y, T: np.sqrt(np.mean(np.sum((Y-T)**2, axis=1)))
calc_cecf = lambda Y, T: np.sum(np.sum(np.vectorize(lambda y, t: -np.log(y) if t==1. else -np.log(1-y))(Y, T), axis=1))
correct_calcs_vector = lambda Y, T: np.sum(np.vectorize(lambda x: 0 if x < 0.5 else 1)(Y).astype(np.uint8)==T.astype(np.uint8), axis=1)==Y.shape[1]
correct_calcs = lambda Y, T: np.sum(correct_calcs_vector(Y, T))
calc_diff = lambda bws, bwsd, eta: map(lambda (i, a, b): a-b*i**1.5*eta, zip(np.arange(len(bws), 0, -1.), bws, bwsd))

def testcase_correct_calcs():
    Y = np.array([[0.4, 0.6], [0.8, 0.3], [0.4, 0.3], [0.8, 0.9]])
    T = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0], [1.0, 1.0]])

    correct = correct_calcs(Y, T)
    print("correct = {}".format(correct))
testcase_correct_calcs()
# raw_input()

def calc_forward(X, bws):
    """
        param
    """
    ones = np.ones((X.shape[0], 1))
    Y = sig(np.hstack((ones, X)).dot(bws[0]))
    for i, bw in enumerate(bws[1:]):
        Y = sig(np.hstack((ones, Y)).dot(bw))
    return Y

def backprop(X, bws, T, reweights):
    Xs = []
    Ys = [X]
    ones = np.ones((X.shape[0], 1))
    A = np.hstack((ones, X)).dot(bws[0]); Xs.append(A)
    Y = sig(A); Ys.append(Y)
    for i, bw in enumerate(bws[1:]):
        A = np.hstack((ones, Y)).dot(bw); Xs.append(A)
        Y = sig(A); Ys.append(Y)
    
    d = (Ys[-1]-T)*sig_dev(Xs[-1])*4
    bwsd = [0 for _ in xrange(len(bws))]
    bwsd[-1] = (np.hstack((ones, Ys[-2]))*reweights).T.dot(d)
    
    for i in xrange(2, len(bws)+1):
        d = d.dot(bws[-i+1][1:].T)*sig_dev(Xs[-i])
        bwsd[-i] = (np.hstack((ones, Ys[-i-1]))*reweights).T.dot(d)*1.25**i

    return bwsd

def numerical_gradient(X, bws, T):
    bwsd = [np.zeros_like(bwsi) for bwsi in bws]

    epsilon = 0.001
    for bwsi, bwsdi in zip(bws, bwsd):
        for y in xrange(0, bwsi.shape[0]):
            for x in xrange(0, bwsi.shape[1]):
                bwsi[y, x] += epsilon
                fr = calc_cecf(calc_forward(X, bws), T)
                bwsi[y, x] -= epsilon*2.
                fl = calc_cecf(calc_forward(X, bws), T)
                bwsi[y, x] += epsilon
                bwsdi[y, x] = (fr - fl) / (2.*epsilon)

    return bwsd

def get_binary_adder_numbers(bits=3):
    def get_binary_numbers(n, bits):
        b = [0 for _ in xrange(bits)]
        i = 1

        while n > 0:
            b[-i] = n%2
            n //= 2
            i += 1

        return b

    get_bin = lambda n, bits: map(lambda x: int(x), bin(n)[2:].zfill(bits))

    max_num = 2**bits
    max_numbers = 2**(2*bits)

    X = np.zeros((max_numbers, 2*bits))
    T = np.zeros((max_numbers, bits+1))

    for i in xrange(max_num):
        for j in xrange(max_num):
            idx = i*max_num+j
            X[idx] = get_bin(i, bits)+get_bin(j, bits)
            T[idx] = get_bin(i+j, bits+1)

    return X, T

bits = 3
# nl = [2*bits, 4*bits, bits+1]
nl = [6, 10, 10, 4]
nlt = [(n1+1, n2) for n1, n2 in zip(nl[:-1], nl[1:])]

# Biases and weights are brought together
bws = [np.random.uniform(-1./nlti[0], 1./nlti[0], nlti) for nlti in nlt] # [np.random.normal(0, 0.02, nlti) for nlti in nlt]
# print("bws.shape = {}".format("".join(map(str, map(lambda x: x.shape, bws)))))

X, T = get_binary_adder_numbers(bits)
# X, T = X[:256], T[:256]

print("X = {}".format(X))
print("T = {}".format(T))

print("X.shape = {}".format(X.shape))
print("T.shape = {}".format(T.shape))

# real_gradient = backprop(X, bws, T)
# numeric_gradient = numerical_gradient(X, bws, T)
# print("backprop =\n{}".format(real_gradient))
# print("numerical gradient =\n{}".format(numeric_gradient))
# print("quotient =\n{}".format(map(lambda (a, b): a/b, zip(real_gradient, numeric_gradient))))
# print("mean of quotient =\n{}".format(
#     np.sum(
#         map(lambda (a, b): np.sum(a/b), zip(real_gradient, numeric_gradient))
#     ) / np.sum(
#         map(lambda a: a.shape[0]*a.shape[1], real_gradient)
#     )))
# raw_input()

print("start learning:")
eta = 0.03
bws_0 = deepcopy(bws)
Y = calc_forward(X, bws)
rmse_0 = calc_rmse(Y, T)
cecf_0 = calc_cecf(Y, T)
reweights = np.ones((X.shape[0], 1))
prev2_deriv = backprop(X, bws, T, reweights)
bws = calc_diff(bws, prev2_deriv, eta)

bws_1 = deepcopy(bws)
Y = calc_forward(X, bws)
rmse_1 = calc_rmse(Y, T)
cecf_1 = calc_cecf(Y, T)

# print("rmse = {:2.8f}".format(rmse_1))
print("cecf = {:2.8f}".format(cecf_1))

Y = calc_forward(X, bws)
cecfs_0 = [calc_cecf(Y, T)]
# cecfs_1 = []
# cecfs_2 = []
# rmses = []
accs = [correct_calcs(Y, T)]
prev_deriv = backprop(X, bws, T, reweights)
calc_deriv_prev_bigger = lambda ps, ds: np.sum(map(lambda (p, d): np.sum(np.abs(p)>np.abs(d)), zip(ps, ds)))
deriv_prev_bigger = [calc_deriv_prev_bigger(prev2_deriv, prev_deriv)]

# calc_length_bws_1st_deriv = lambda bws1, bws0: np.sqrt(np.sum(map(lambda (a, b): np.sum((a-b)**2), zip(bws1, bws0))))
# calc_length_bws_2nd_deriv = lambda bws2, bws1, bws0: np.sqrt(np.sum(map(lambda (a, b, c): np.sum((a-2*b+c)**2), zip(bws2, bws1, bws0))))

max_eta = 10000.
min_eta = 0.000005
# should_increase_learn_rate = True

amount_params = np.sum(map(lambda a: np.prod(a.shape), bws))
print("amount_params = {}".format(amount_params))
# raw_input()
max_iters = 20000
weights_history = np.zeros((max_iters, amount_params))
accs_history = np.zeros((max_iters, T.shape[0]))

# shifter = -1
all_correct_iter = 0
skip = 0
# Look for LASSO!!!
bigger_counter = 0
for i in xrange(0, max_iters):
    if i >= max_iters:
        break

    # bws = map(lambda x: x+0.0001/(i+1)**1.1, bws)
    
    # bws = map(lambda x: x+np.random.normal(0, 1./(1000+i), x.shape), bws)
    # if i % 1000 == 0:
    #     shifter *= -1
    bwsd = backprop(X, bws, T, reweights)
    deriv_prev_bigger.append(calc_deriv_prev_bigger(prev_deriv, bwsd))
    temp = bwsd
    def reweight_derivs(p, d):
        pa, da = np.abs(p), np.abs(d)
        s = np.zeros_like(p)
        # print("p =\n{}".format(p))
        # print("d =\n{}".format(d))
        bigger = np.where(pa > da)
        smaller = np.where(pa <= da)
        # print("bigger =\n{}".format(bigger))
        s[bigger] = d[bigger]*0.25
        s[smaller] = d[smaller]*1.0
        # print("s =\n{}".format(s))
        # raw_input()
        return s
    # bwsd = map(lambda (p, d): reweight_derivs(p, d), zip(prev_deriv, bwsd))

    def reweight_derivs_2(p2s, ps, ds):
        # if np.abs(deriv_prev_bigger[-2] - deriv_prev_bigger[-1]) > 20:
        if deriv_prev_bigger[-1] > 30:
            # print("dev[-2] dev[-1] > 100 is TRUE!")
            return map(lambda (p2, p, d): d-p*0.5+p2*0.25, zip(p2s, ps, ds))
        else:
            return map(lambda (p2, p, d): d-p*0.125+p2*0.0625, zip(p2s, ps, ds))
            # return map(lambda (p2, p, d): d-p*0.25+p2*0.125, zip(p2s, ps, ds))
        # scale = float(amount_params - deriv_prev_bigger[-1]) / amount_params / 2.5
        # return map(lambda (p2, p, d): d-p*scale*2.+p2*scale*1., zip(p2s, ps, ds))
    # bwsd = reweight_derivs_2(prev2_deriv, prev_deriv, bwsd)
    bwsd = map(lambda (p2, p, d): d-p*0.5+p2*0.25, zip(prev2_deriv, prev_deriv, bwsd))
    # bwsd = map(lambda (p2, p, d): d-p*0.5+p2*0.25, zip(prev2_deriv, prev_deriv, bwsd))

    prev2_deriv = prev_deriv
    prev_deriv = temp
    etaplus = eta*1.02
    etaminus = eta/1.1

    bwsp = calc_diff(bws, bwsd, etaplus)
    bwsm = calc_diff(bws, bwsd, etaminus)
    # bwsp[:1] = bws[:1]
    # bwsm[:1] = bws[:1]

    cecf_p = calc_cecf(calc_forward(X, bwsp), T)
    cecf_2 = calc_cecf(calc_forward(X, bws), T)
    cecf_m = calc_cecf(calc_forward(X, bwsm), T)

    if cecf_p < cecf_2:
        last_if = 1
        # neweta = eta / 1.2
        # for j in xrange(0, 20):
        #     neww_ /= 1.2
        #     bwsn = calc_diff(bws, bwsd, neweta)
        #     cecf_n = calc_cecf(calc_forward(X, bwsn), T)
        #     if cecf_n < cecf_2:
        #         cecf_2 = cecf_n
        #         break
        cecf_2 = cecf_p
        bws = bwsp
        skip += 1
        if skip > 50:# 10:
            skip = 0
            eta = max_eta if etaplus > max_eta else etaplus
        eta = max_eta if etaplus > max_eta else etaplus
    elif cecf_m > cecf_2:
        last_if = 2
        skip = 0
        neweta = eta / 1.05
        for j in xrange(0, 5):
            neweta /= 1.2
            bwsn = calc_diff(bws, bwsd, neweta)
            if neweta < min_eta:
                neweta = min_eta
                # break
            # bwsn[:1] = bws[:1]
            cecf_n = calc_cecf(calc_forward(X, bwsn), T)
            if cecf_n < cecf_2:
                cecf_2 = cecf_n
                break
        eta = neweta
        bws = bwsn
    else:
        last_if = 3
        skip = 0
        cecf_2 = cecf_m
        eta = etaminus
        bws = bwsm
    # bws_2 = bws
    # first_deriv = (cecf_2-cecf_1)/calc_length_bws_1st_deriv(bws_2, bws_1)
    # second_deriv = (cecf_2-2*cecf_1+cecf_0)/calc_length_bws_2nd_deriv(bws_2, bws_1, bws_0)

    # if should_increase_learn_rate:
    #     eta *= 1.001
    #     eta = max_eta if eta > max_eta else eta
    #     if second_deriv < -0.1:
    #         should_increase_learn_rate = False
    # else:
    #     eta /= 1.01
    #     eta = min_eta if eta < min_eta else eta
    #     if second_deriv > 0.1:
    #         should_increase_learn_rate = True

    cecfs_0.append(cecf_2)
    # cecfs_1.append(first_deriv)
    # cecfs_2.append(second_deriv)
    
    # cecf_0 = cecf_1
    # cecf_1 = cecf_2
    # bws_0 = bws_1
    # bws_1 = bws_2

    # rmse = calc_rmse(Y, T)
    # rmses.append(rmse)

    Y = calc_forward(X, bws)
    corr_calcs = correct_calcs(Y, T)
    accs.append(corr_calcs)

    predictions_bool = correct_calcs_vector(Y, T)
    reweights[:] = 1.
    reweights[np.where(predictions_bool == False)] = 1.1
    # print("prediction_bool: {}".format(predictions_bool))
    # print("reweights: {}".format(reweights))
    # raw_input("Press ENTER...")

    # Insert the bwsd in the weights_histroy variable
    weights_history[i] = np.hstack(map(lambda a: a.flatten(), bwsd))
    accs_history[i] = correct_calcs_vector(Y, T)

    # if i%100 == 0:
    #     eta *= 1.02
    print("epoch: {}, eta: {:1.6f}, cecf: {:5.8f}, acc: {:3}, amount(p>d) = {}".format(i, eta, cecf_2, corr_calcs, deriv_prev_bigger[-1]))

    if cecfs_0[-2] > cecfs_0[-1] and (accs[-1] == 2**(bits*2)): # or cecfs_0[-2] - cecfs_0[-1] < 0.01):
        all_correct_iter += 1
        # if all_correct_iter > 10:
        #     break
    else:
        all_correct_iter = 0

    if cecfs_0[-1] > cecfs_0[-2]:
        if bigger_counter > 100:
            print("last_if: {}".format(last_if))
            break
    else:
        bigger_counter = 0

    # User break the loop by pressing ENTER!
    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        break

# Save the pixel picture!
first_column = 5
I_bws = np.zeros((max_iters, first_column+amount_params, 3)) # for RGB
def get_condition_values_only(X, pos, times):
    Xn = np.zeros_like(X)
    Xn[pos] = times*np.sqrt(np.abs(X[pos]))
    return Xn
X_p = get_condition_values_only(weights_history, np.where(weights_history>=0.), 1.)
X_m = -get_condition_values_only(weights_history, np.where(weights_history<0.), -1.)

X_p = X_p/np.max(X_p)*255
X_m = X_m/np.max(X_m)*255

rows_indicator = 100
for i in xrange(0, max_iters//rows_indicator):
    I_bws[rows_indicator*i:rows_indicator*(i+1), :first_column-1] = [128*(i%2), 128, 128*((i+1)%2)]

# I_bws[:, first_column-1:first_column, :] = 255
# I_bws[:, first_column:, 0] = X_p
# I_bws[:, first_column:, 1] = 2
# I_bws[:, first_column:, 2] = X_m
# img_bws = Image.fromarray(I_bws.transpose(1, 0, 2).astype(np.uint8))
# img_bws.save("bias_weights_derivations.png", "PNG")

# I_accs = np.zeros((max_iters, accs_history.shape[1], 3))
# I_accs[:, :, 0] = accs_history * 255
# I_accs[:, :, 1] = accs_history * 255
# I_accs[:, :, 2] = accs_history * 255
# img_accs = Image.fromarray(I_accs.transpose(1, 0, 2).astype(np.uint8))
# img_accs.save("accs_history.png", "PNG")

# plt.figure()
# plt.title("RMSE error")
# plt.xlabel("epoch")
# plt.ylabel("error")
# plt.plot(np.arange(len(rmses)), rmses, "b-")

plt.figure()
plt.title("CECF error function")
plt.xlabel("epoch")
plt.ylabel("error")
plt.plot(np.arange(len(cecfs_0)), cecfs_0, "b-")
plt.savefig("cecf_error_function.png", format="png", dpi=400)

# plt.figure()
# plt.title("CECF error function 1st derivation")
# plt.xlabel("epoch")
# plt.ylabel("error")
# plt.plot(np.arange(len(cecfs_1)), cecfs_1, "b-")

# plt.figure()
# plt.title("CECF error function 2nd derivation")
# plt.xlabel("epoch")
# plt.ylabel("error")
# plt.plot(np.arange(len(cecfs_2)), cecfs_2, "b-")

plt.figure()
plt.title("Accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.plot(np.arange(len(accs)), accs, "b-")
plt.savefig("accuracy.png", format="png", dpi=400)

plt.figure()
plt.title("Deriv prev bigger")
plt.xlabel("epoch")
plt.ylabel("amount of deriv prev bigger")
plt.plot(np.arange(len(deriv_prev_bigger)), deriv_prev_bigger, "b-")
plt.savefig("deriv_prev_bigger.png", format="png", dpi=400)

# TODO add etha plot

# raw_input("Finished!")
# plt.show()
