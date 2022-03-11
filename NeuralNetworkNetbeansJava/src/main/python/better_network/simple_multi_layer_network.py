#! /usr/bin/python2.7

import csv
import gzip
import os
import select
import sys
import time

import numpy as np
import multiprocessing as mp
import pickle as pkl

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr

from copy import deepcopy
from PIL import Image
from UtilsBinary import *

sig = lambda X: 1. / (1 + np.exp(-X))
sig_dev = lambda X: sig(X) * (1 - sig(X))
relu = lambda x: np.vectorize(lambda x: 0. if x < 0. else x*0.1)(x)
relu_dev = lambda x: np.vectorize(lambda x: 0. if x < 0. else 0.001)(x)
tanh = lambda x: np.tanh(x)
tanh_dev = lambda x: 1 - np.tanh(x)**2

f_name = "tanh"
f = tanh
fp = tanh_dev

calc_rmse = lambda Y, T: np.sqrt(np.mean(np.sum((Y-T)**2, axis=1)))
calc_cecf = lambda Y, T: np.sum(np.sum(np.vectorize(lambda y, t: -np.log(y) if t==1. else -np.log(1-y))(Y, T), axis=1))
correct_calcs_vector = lambda Y, T: np.sum(np.vectorize(lambda x: 0 if x < 0.5 else 1)(Y).astype(np.uint8)==T.astype(np.uint8), axis=1)==Y.shape[1]
correct_calcs = lambda Y, T: np.sum(correct_calcs_vector(Y, T))
calc_diff = lambda bws, bwsd, eta: map(lambda (i, a, b): a-b*i**1.5*eta, zip(np.arange(len(bws), 0, -1.), bws, bwsd))
calc_deriv_prev_bigger = lambda ps, ds: np.sum(map(lambda (p, d): np.sum(np.abs(p)>np.abs(d)), zip(ps, ds)))
calc_deriv_prev_bigger_per_layer = lambda ps, ds: map(lambda (p, d): np.sum(np.abs(p)>np.abs(d)), zip(ps, ds))

def testcase_correct_calcs():
    Y = np.array([[0.4, 0.6], [0.8, 0.3], [0.4, 0.3], [0.8, 0.9]])
    T = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0], [1.0, 1.0]])

    correct = correct_calcs(Y, T)
    print("correct = {}".format(correct))

def calc_forward(X, bws):
    """
        param
    """
    ones = np.ones((X.shape[0], 1))
    Y = f(np.hstack((ones, X)).dot(bws[0]))
    for i, bw in enumerate(bws[1:-1]):
        Y = f(np.hstack((ones, Y)).dot(bw))
    Y = sig(np.hstack((ones, Y)).dot(bws[-1]))
    return Y

def backprop(X, bws, T, reweights):
    Xs = []
    Ys = [X]
    ones = np.ones((X.shape[0], 1))
    A = np.hstack((ones, X)).dot(bws[0]); Xs.append(A)
    Y = f(A); Ys.append(Y)
    for i, bw in enumerate(bws[1:-1]):
        A = np.hstack((ones, Y)).dot(bw); Xs.append(A)
        Y = f(A); Ys.append(Y)
    A = np.hstack((ones, Y)).dot(bws[-1]); Xs.append(A)
    Y = sig(A); Ys.append(Y)
    
    d = (Ys[-1]-T)*sig_dev(Xs[-1])*4#*fp(Xs[-1])*4
    bwsd = [0 for _ in xrange(len(bws))]
    bwsd[-1] = (np.hstack((ones, Ys[-2]))*reweights).T.dot(d)
    
    for i in xrange(2, len(bws)+1):
        d = d.dot(bws[-i+1][1:].T)*fp(Xs[-i])
        bwsd[-i] = (np.hstack((ones, Ys[-i-1]))*reweights).T.dot(d)*1.3**i

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

class NeuralNetwork(Exception):
    len_train = None
    len_test = None

    reweight_factor = 1.2
    epochs = 0

    bws = None

    bits = None
    nl = None
    f_name = None

    max_eta = 10000.
    min_eta = 0.000005

    max_iters = 5000
    eta = 0.0001
    first_eta = eta
    plus_factor = 1.002
    minus_factor = 1.1
    first_deriv_factor = 0.5
    second_deriv_factor = 0.5

    prev2_deriv = None
    prev_deriv = None

    get_bws = lambda self, nl: [np.random.uniform(-1./n1, 1./n1, (n1+1, n2)) for n1, n2 in zip(nl[:-1], nl[1:])]

    def __init__(self, nl=None):
        self.cecfs_0_train = []
        self.cecfs_0_test = []
        self.accs = []
        self.accs_train = []
        self.accs_test = []
        self.misclass_train = [] # in %
        self.misclass_test = [] # in %
        self.deriv_prev_bigger = []
        self.deriv_prev_bigger_per_layer = []
        self.etas = []

        if nl != None:
            self.set_nl(nl)

    def set_nl(self, nl):
        self.nl = nl
        self.bws = self.get_bws(nl)

    def fit(self, X_train, X_test, T_train, T_test):
        self.len_train = len(X_train)
        self.len_test = len(X_test)

        bws = self.bws
        max_iters = self.max_iters
        eta = self.eta

        max_eta = self.max_eta
        min_eta = self.min_eta

        plus_factor = self.plus_factor
        minus_factor = self.minus_factor

        first_deriv_factor = self.first_deriv_factor
        second_deriv_factor = self.second_deriv_factor

        reweights = np.ones((X_train.shape[0], 1))
        if self.epochs == 0:
            prev2_deriv = backprop(X_train, bws, T_train, reweights)
            bws = calc_diff(bws, prev2_deriv, eta)
            prev_deriv = backprop(X_train, bws, T_train, reweights)
            bws = calc_diff(bws, prev_deriv, eta)
        else:
            prev2_deriv = self.prev2_deriv
            prev_deriv = self.prev_deriv
        Y_train = calc_forward(X_train, bws)
        Y_test = calc_forward(X_test, bws)

        cecfs_0_train = [calc_cecf(Y_train, T_train)]
        cecfs_0_test = [calc_cecf(Y_test, T_test)]
        accs_train = [correct_calcs(Y_train, T_train)]
        accs_test = [correct_calcs(Y_test, T_test)]
        misclass_train = [1-accs_train[0]/float(self.len_train)]
        misclass_test = [1-accs_test[0]/float(self.len_test)]
        deriv_prev_bigger = [calc_deriv_prev_bigger(prev2_deriv, prev_deriv)]
        deriv_prev_bigger_per_layer = [calc_deriv_prev_bigger_per_layer(prev2_deriv, prev_deriv)]
        etas = [eta]

        amount_params_all = np.sum(map(lambda a: np.prod(a.shape), bws))
        amount_params_per_layer = map(lambda a: np.prod(a.shape), bws)
        print("amount_params_all: {}".format(amount_params_all))
        print("amount_params_per_layer: {}".format(amount_params_per_layer))
        weights_history = np.zeros((max_iters, amount_params_all))
        accs_history_train = np.zeros((max_iters, T_train.shape[0]))
        accs_history_test = np.zeros((max_iters, T_test.shape[0]))

        all_correct_iter = 0
        # skip = 0
        bigger_counter = 0

        # Look for LASSO!!!
        for epoch in xrange(0, max_iters):
            if epoch >= max_iters:
                break

            predictions_bool = correct_calcs_vector(Y_train, T_train)
            reweights[:] = 1.
            reweights[np.where(predictions_bool == False)] = self.reweight_factor

            bwsd = backprop(X_train, bws, T_train, reweights)
            deriv_prev_bigger.append(calc_deriv_prev_bigger(prev_deriv, bwsd))
            deriv_prev_bigger_per_layer.append(calc_deriv_prev_bigger_per_layer(prev_deriv, bwsd))
            temp = bwsd
            def reweight_derivs(p, d):
                pa, da = np.abs(p), np.abs(d)
                s = np.zeros_like(p)
                bigger = np.where(pa > da)
                smaller = np.where(pa <= da)
                s[bigger] = d[bigger]*0.25
                s[smaller] = d[smaller]*1.0
                return s

            def reweight_derivs_2(p2s, ps, ds):
                if deriv_prev_bigger[-1] > 30:
                    return map(lambda (p2, p, d): d-p*0.5+p2*0.25, zip(p2s, ps, ds))
                else:
                    return map(lambda (p2, p, d): d-p*0.125+p2*0.0625, zip(p2s, ps, ds))
            # bwsd = reweight_derivs_2(prev2_deriv, prev_deriv, bwsd)
            bwsd = map(lambda (p2, p, d): d+p*first_deriv_factor+p2*second_deriv_factor, zip(prev2_deriv, prev_deriv, bwsd))
            # bwsd = map(lambda (p2, p, d): d-p*0.5+p2*0.25, zip(prev2_deriv, prev_deriv, bwsd))

            prev2_deriv = prev_deriv
            prev_deriv = temp
            etaplus = eta*plus_factor
            etaminus = eta/minus_factor

            bwsp = calc_diff(bws, bwsd, etaplus)
            bwsm = calc_diff(bws, bwsd, etaminus)

            cecf_p = calc_cecf(calc_forward(X_train, bwsp), T_train)
            cecf_0_train = calc_cecf(calc_forward(X_train, bws), T_train)
            cecf_m = calc_cecf(calc_forward(X_train, bwsm), T_train)

            if cecf_p < cecf_0_train:
                last_if = 1
                cecf_0_train = cecf_p
                bws = bwsp
                eta = max_eta if etaplus > max_eta else etaplus
            else:
                last_if = 2
                cecf_0_train = cecf_m
                bws = bwsm
                eta = min_eta if etaminus < min_eta else etaminus

            cecfs_0_train.append(cecf_0_train)

            Y_train = calc_forward(X_train, bws)
            corr_calcs_train = correct_calcs(Y_train, T_train)
            accs_train.append(corr_calcs_train)
            misclass_train.append(1-corr_calcs_train/float(self.len_train))

            cecf_0_test = calc_cecf(calc_forward(X_test, bws), T_test)
            cecfs_0_test.append(cecf_0_test)

            Y_test = calc_forward(X_test, bws)
            corr_calcs_test = correct_calcs(Y_test, T_test)
            accs_test.append(corr_calcs_test)
            misclass_test.append(1-corr_calcs_test/float(self.len_test))

            # Insert the bwsd in the weights_histroy variable
            weights_history[epoch] = np.hstack(map(lambda a: a.flatten(), bwsd))
            accs_history_train[epoch] = correct_calcs_vector(Y_train, T_train)
            accs_history_test[epoch] = correct_calcs_vector(Y_test, T_test)
            etas.append(eta)

            # print("epoch: {}, eta: {:1.6f}, cecf train: {:5.5f}, mis train: {:2.3f}%, cecf test: {:5.5f}, mis test: {:2.3f}%, amount(p>d) = {}".format(
            #     epoch, eta, cecf_0_train, (1.-corr_calcs_train/float(len(X_train)))*100., cecf_0_test, (1.-corr_calcs_test/float(len(X_test)))*100., deriv_prev_bigger[-1]))

            if cecfs_0_train[-2] > cecfs_0_train[-1] and (accs_test[-1] == len(X_test)): # or cecfs_0[-2] - cecfs_0[-1] < 0.01):
                all_correct_iter += 1
                if all_correct_iter > 50:
                    break
            else:
                all_correct_iter = 0

            if cecfs_0_train[-1] > cecfs_0_train[-2] or cecfs_0_train[-2] - cecfs_0_train[-1] < 0.0001:
                bigger_counter += 1
                if bigger_counter > 100:
                    print("last_if: {}".format(last_if))
                    break
            else:
                bigger_counter = 0

        self.prev2_deriv = prev2_deriv
        self.prev_deriv = prev_deriv
        self.bws = bws
        self.cecfs_0_train.extend(cecfs_0_train[1:])
        self.cecfs_0_test.extend(cecfs_0_test[1:])
        self.accs_train.extend(accs_train[1:])
        self.accs_test.extend(accs_test[1:])
        self.misclass_train.extend(misclass_train[1:])
        self.misclass_test.extend(misclass_test[1:])
        self.deriv_prev_bigger.extend(deriv_prev_bigger[1:])
        self.deriv_prev_bigger_per_layer.extend(deriv_prev_bigger_per_layer[1:])
        self.etas.extend(etas[1:])
        self.eta = eta
        self.epochs += epoch

    def get_plots(self, infix1="", infix2="", suffix="", plots_folder_name=None):
        bws = self.bws

        amount_params_all = np.sum(map(lambda a: np.prod(a.shape), bws))
        amount_params_per_layer = map(lambda a: np.prod(a.shape), bws)

        fig, axarr = plt.subplots(4, 1, figsize=(8, 16))

        major_ticks = 1000 if self.epochs < 7500 else 5000
        minor_ticks = 100  if self.epochs < 7500 else 500
        def set_axis_properties(ax):
            ax.set_xlim([0, self.epochs])
            ax.xaxis.set_major_locator(tkr.MultipleLocator(major_ticks))
            ax.xaxis.set_minor_locator(tkr.MultipleLocator(minor_ticks))
            for xmin in ax.xaxis.get_minorticklocs():
                if xmin % major_ticks == 0:
                    continue
                ax.axvline(x=xmin, ls="--", color="g", linewidth=0.5)

        axarr[0].set_title("CECF error function")
        axarr[0].set_xlabel("epoch")
        axarr[0].set_ylabel("error")
        axarr[0].plot(np.arange(len(self.cecfs_0_train)), self.cecfs_0_train, "b-")
        axarr[0].plot(np.arange(len(self.cecfs_0_test)), self.cecfs_0_test, "g-")
        axarr[0].grid(True)
        set_axis_properties(axarr[0])

        axarr[1].set_title("Misclassification")
        axarr[1].set_xlabel("epoch")
        axarr[1].set_ylabel("misclass. [%]")
        axarr[1].plot(np.arange(len(self.accs_train)), (1-np.array(self.accs_train)/float(self.len_train))*100, "b-")#, linewidth=0.1)
        axarr[1].plot(np.arange(len(self.accs_test)), (1-np.array(self.accs_test)/float(self.len_test))*100, "g-")#, linewidth=0.1)
        axarr[1].set_ylim([0., 100.])
        axarr[1].grid(True)
        set_axis_properties(axarr[1])

        axarr[2].set_title("Deriv prev bigger per layer")
        axarr[2].set_xlabel("epoch")
        axarr[2].set_ylabel("amount of deriv prev bigger")
        axarr[2].set_ylim([0., 1.])
        axarr[2].grid(True)
        set_axis_properties(axarr[2])

        axarr[3].set_title("Eta per epoch")
        axarr[3].set_xlabel("epoch")
        axarr[3].set_ylabel("eta value")
        axarr[3].set_yscale("log")
        axarr[3].plot(np.arange(len(self.etas)), self.etas, "b-", linewidth=0.5)
        axarr[3].grid(True)
        set_axis_properties(axarr[3])

        plt.tight_layout()

        deriv_prev_bigger_per_layer = (np.array(self.deriv_prev_bigger_per_layer) / map(float, amount_params_per_layer)).T
        plots = []
        for dpbpl in deriv_prev_bigger_per_layer:
            plots.append(axarr[2].plot(np.arange(len(dpbpl)), dpbpl, linewidth=0.2)[0])
            # axarr[2].plot(np.arange(len(self.deriv_prev_bigger)), self.deriv_prev_bigger, "b-", linewidth=0.2)
        labels = ("bw layer {}".format(i+1) for i in range(0, len(deriv_prev_bigger_per_layer)))
        axarr[2].legend(plots, labels, fontsize=6, bbox_to_anchor=(1.005, 1.13))

        fig.subplots_adjust(top=0.92)
        nl_str = "_".join(map(str, self.nl))
        fig.suptitle("split_rate: {:1.2f}, bits: {}, nl: {}, epochs: {}, first eta: {:1.5f}\nplus factor: {:1.5f}, minus factor: {:1.5f}\n first deriv factor: {:1.3f}, second deriv factor: {:1.3f}\ncecf train: {:1.3f}, cecf test: {:1.3f}, mis train: {:1.3f}%, mis test: {:.3f}%".format(
            self.split_rate, self.bits, nl_str, self.epochs, self.first_eta, self.plus_factor, self.minus_factor, self.first_deriv_factor, self.second_deriv_factor,
            self.cecfs_0_train[-1], self.cecfs_0_test[-1], (1-self.accs_train[-1]/float(self.len_train))*100, (1-self.accs_test[-1]/float(self.len_test))*100), fontsize=12)

        fig.savefig((plots_folder_name+"/" if plots_folder_name != None else "")+"all_plots_in_one{}_bits_{}{}_nl_{}{}.png".format(infix1, self.bits, infix2, nl_str, suffix), format="png", dpi=500)

def process_task(input_queue, output_queue):
    proc_num = None
    nn_file_name = None
    suffix = None
    split_rate = None
    X_train = None
    T_train = None
    X_test = None
    T_test = None
    plots_folder_name = None

    while True:
        command, args = input_queue.get()

        if command == "FINISHED":
            break
        elif command == "SET_PARAMS":
            proc_num = args[0]
            nn_file_name, suffix, split_rate, X_train, T_train, X_test, T_test, plots_folder_name = args[1]

            print("proc_num: {}".format(proc_num))
        elif command == "FIT_NN":
            print("STARTED FIT_NN proc_num {}".format(proc_num))
            with gzip.open(nn_file_name, "rb") as fin:
                nn = pkl.load(fin)
            nn.fit(X_train, X_test, T_train, T_test)
            # nn.get_plots(suffix=suffix, plots_folder_name=plots_folder_name)
            
            # os.remove(nn_file_name)
            with gzip.open(nn_file_name, "wb") as fout:
                pkl.dump(nn, fout)
            print("FINISHED FIT_NN proc_num {}".format(proc_num))
            output_queue.put((nn.nl, split_rate, nn.misclass_train[-1]*100, nn.misclass_test[-1]*100, nn.epochs))
            # output_queue.put(("_".join(map(str, nn.nl)), str(split_rate), str(nn.epochs), str(nn.misclass_train[-1]), str(nn.misclass_test[-1])))

def get_train_test_sets_splitted(bits, split_rates, suffix=""):
    X, T = get_binary_adder_numbers(bits)

    perms = np.random.permutation(np.arange(0, len(X)))
    X = X[perms]
    T = T[perms]

    if not os.path.exists("train_test_sets"):
        os.makedirs("train_test_sets")

    train_test_sets_splitted = []
    for split_rate in split_rates:
        split = int(len(X)*split_rate)
        X_train = X[:split]
        T_train = T[:split]
        X_test = X[split:]
        T_test = T[split:]

        train_test_set_file_name = "train_test_sets/train_test_set_bin_adder_{}_bits_split_rate_{}{}.pkl.gz".format(bits, str(split_rate).replace(".", "_"), suffix)
        
        # if not os.path.exists(train_test_set_file_name):
        #     with gzip.open(train_test_set_file_name, "wb") as fout:
        #         pkl.dump((X_train, T_train, X_test, T_test), fout)
        # else:
        #     with gzip.open(train_test_set_file_name, "rb") as fin:
        #         X_train, T_train, X_test, T_test = pkl.load(fin)
        with gzip.open(train_test_set_file_name, "wb") as fout:
            pkl.dump((X_train, T_train, X_test, T_test), fout)
        
        train_test_sets_splitted.append((split_rate, X_train, T_train, X_test, T_test))

    return train_test_sets_splitted

def get_args(train_test_sets_splitted, network_folder_name, bits=3, neural_tests=5, neural_jumps=1, try_nums=3):
    args = []
    for split_rate, X_train, T_train, X_test, T_test in train_test_sets_splitted:
        for j in xrange(0, neural_tests):
            for try_num in xrange(0, try_nums):
                nl[1] = bits+1+neural_jumps*j
                nn = NeuralNetwork(nl)

                nn.bits = bits
                nn.nl = nl
                nn.f_name = f_name
                nn.split_rate = split_rate

                split_rate_str = str(split_rate).replace(".", "_")
                nn_file_name = network_folder_name+"/nn_{}_try_{}_split_rate_{}.pkl.gz".format("_".join(map(str, nl)), try_num+1, split_rate_str)
                with gzip.open(nn_file_name, "wb") as fout:
                    pkl.dump(nn, fout)
                args.append((nn_file_name, "_try_{}_split_rate_{}".format(try_num+1, split_rate_str), split_rate, X_train, T_train, X_test, T_test, plots_folder_name))

    return args

def do_all_args(args):
    max_amount_procs = int(mp.cpu_count()*0.85)
    input_queues = [mp.Queue() for _ in xrange(0, max_amount_procs)]
    output_queues = [mp.Queue() for _ in xrange(0, max_amount_procs)]
    procs = [mp.Process(target=process_task, args=(input_queues[i], output_queues[i])) for i in xrange(0, max_amount_procs)]
    for proc in procs: proc.start()
    procs_nums = list(xrange(0, len(args)))
    running_procs = [False for _ in xrange(0, max_amount_procs)]

    results = []

    # TODO: do first m nns
    for i in xrange(0, max_amount_procs):
        if len(procs_nums) == 0:
            break
        proc_num = procs_nums.pop(0)
        arg = args.pop(0)
        input_queues[i].put(("SET_PARAMS", (proc_num, arg)))
        input_queues[i].put(("FIT_NN", (0, )))
        running_procs[i] = True
    # TODO: do n-m nns
    while len(procs_nums) > 0:
        for in_q, out_q, proc in zip(input_queues, output_queues, procs):
            ret = None
            try:
                ret = out_q.get_nowait()
            except:
                pass
            if ret != None:
                # statistics_csv.writerow(ret)
                results.append(ret)
                proc_num = procs_nums.pop(0)
                arg = args.pop(0)
                in_q.put(("SET_PARAMS", (proc_num, arg)))
                in_q.put(("FIT_NN", (0, )))
    # TODO: do last m nns
    while np.sum(running_procs) > 0:
        for i, (in_q, out_q, proc) in enumerate(zip(input_queues, output_queues, procs)):
            ret = None
            try:
                ret = out_q.get_nowait()
            except:
                pass
            if ret != None:
                results.append(ret)
                running_procs[i] = False

    for input_queue in input_queues: input_queue.put(("FINISHED", (0,)))
    for proc in procs: proc.join()

    return results

def save_results_dict(nls, split_rates_res, misclasses_train, misclasses_test, epochs, suffix=""):
    results_dict = {}
    results_dict["nls"] = nls
    results_dict["split_rates_res"] = split_rates_res
    results_dict["misclasses_train"] = misclasses_train
    results_dict["misclasses_test"] = misclasses_test
    results_dict["epochs"] = epochs
    if not os.path.exists("results"):
        os.makedirs("results")
    with gzip.open("results/results_dict{}.pkl.gz".format(suffix), "wb") as fout:
        pkl.dump(results_dict, fout)

def save_results_to_csv_file(bits, nls_str, split_rates_str, misclasses_train_str, misclasses_test_str, epochs_str, suffix=""):
    if not os.path.exists("statistics"):
        os.makedirs("statistics")
    csv_file = open("statistics/statistics_binary_adder_{}_bits{}.csv".format(bits, suffix), "wb")
    statistics_csv = csv.writer(csv_file, delimiter=";")

    space_tuples = tuple(("" for _ in xrange(0, len(split_rates_str)-1)))
    statistics_csv.writerow(("nl", "misclass train average")+space_tuples+
                            ("nl", "misclass test average")+space_tuples+
                            ("nl", "epochs average")+space_tuples)

    statistics_csv.writerow(("",)+split_rates_str+("",)+split_rates_str+("",)+split_rates_str)

    for nl, misclass_train, misclass_test, epoch in zip(nls_str, misclasses_train_str, misclasses_test_str, epochs_str):
        statistics_csv.writerow([nl]+misclass_train+[nl]+misclass_test+[nl]+epoch)

    csv_file.close()

colors = ["b", "g", "r", "k", "c", "m", "silver", "firebrick", "darkorchid", "purple", "darkgreen", "y"]

def get_bars_plot(bits, try_nums, rates, split_rates_str, is_epochs=False, infix="", suffix=""):
    # TODO: make also a plot with points!
    ind = np.arange(len(rates))
    width_complete = 0.75
    width_one_bar = width_complete / float(len(rates[0]))

    fig, ax = plt.subplots(figsize=(10, 6))

    rects = []
    for i, rate in enumerate(zip(*rates)):
        rects.append(ax.bar(ind-width_complete/2.+width_one_bar*i, rate, width_one_bar, color=colors[i]))

    ax.set_xlim([-0.5, len(rates)-0.5])
    ax.get_yaxis().get_major_formatter().set_scientific(False)
    if not is_epochs:
        ax.set_ylim([0., 100.])
        ax.set_ylabel("misclass rate [%]")
    else:
        ax.set_ylabel("epochs")
    ax.set_title("binary adder {} bits, misclass rate{}, average of {} tries".format(bits, infix, try_nums))
    ax.set_xlabel("architectures")
    ax.set_xticks(np.arange(0, len(nls_str), 1))
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    ax.set_xticklabels(nls_str) # [""]+nls_str)
    ax.grid(True)

    plt.tight_layout()
    plt.subplots_adjust(right=0.89) # , left=0.1, top=0.9, bottom=0.1)
    ax.legend(rects, split_rates_str, bbox_to_anchor=(1, 1), loc="upper left", fontsize=12) # , loc='upper right', ncol=1) #, loc=(0.8, 0.5))
    if not os.path.exists("bar_plots"):
        os.makedirs("bar_plots")
    plt.savefig("bar_plots/bar_plots_binary_adder_{}_bits{}.png".format(bits, suffix), format="png", dpi=500)

def get_full_bar_plot(bits, try_nums, misclass_train_rates, misclass_test_rates, epochs_rates, nls_str, split_rates_str, suffix=""):
    # TODO: make also a plot with points!
    ind = np.arange(len(misclass_train_rates))
    width_complete = 0.75
    width_one_bar = width_complete / float(len(misclass_train_rates[0]))

    fig, ax = plt.subplots(3, 1, figsize=(10, 18))

    def set_ax(ax, rates, title="", is_epochs=False):
        rects = []
        for i, rate in enumerate(zip(*rates)):
            rects.append(ax.bar(ind-width_complete/2.+width_one_bar*i, rate, width_one_bar, color=colors[i]))

        ax.set_xlim([-0.5, len(rates)-0.5])
        ax.get_yaxis().get_major_formatter().set_scientific(False)
        if not is_epochs:
            ax.set_ylim([0., 100.])
            ax.set_ylabel("misclass rate [%]")
        else:
            ax.set_ylabel("epochs")
        ax.set_title(title)
        ax.set_xlabel("architectures")
        ax.set_xticks(np.arange(0, len(nls_str), 1))
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        ax.set_xticklabels(nls_str)
        ax.grid(True)
        ax.legend(rects, split_rates_str, bbox_to_anchor=(1, 1), loc="upper left", fontsize=12, title="split rate for\ntrain set")

    plt.tight_layout()
    plt.subplots_adjust(top=0.94, left=0.07, bottom=0.05, right=0.87, hspace=0.26)
    
    plt.suptitle("Binary adder for {} bits numbers, number of tries: {}".format(bits, try_nums), fontsize=16)
    set_ax(ax[0], misclass_train_rates, title="Misclass rates for train set")
    set_ax(ax[1], misclass_test_rates, title="Misclass rates for test set ")
    set_ax(ax[2], epochs_rates, title="Taken epochs for learning", is_epochs=True)

    if not os.path.exists("bar_plots"):
        os.makedirs("bar_plots")
    plt.savefig("bar_plots/bar_plots_binary_adder_{}_bits{}.png".format(bits, suffix), format="png", dpi=500)

bits = 2
nl = [bits*2, bits*2, bits+1]
suffix = ""
if len(sys.argv) >= 2:
    bits = int(sys.argv[1])
    nl = [bits*2, bits*2, bits+1]
if len(sys.argv) >= 3:
    suffix = "_"+sys.argv[2]

print("sys.argv: {}".format(sys.argv))
print("len(sys.argv): {}".format(len(sys.argv)))
print("suffix: {}".format(suffix))
print("bits: {}".format(bits))
print("nl: {}".format(nl))

plots_folder_name = "plots/plots_{}_bits{}".format(bits, suffix)
network_folder_name = "networks/networks_{}_bits{}".format(bits, suffix)

if not os.path.exists(plots_folder_name):
    os.makedirs(plots_folder_name)
if not os.path.exists(network_folder_name):
    os.makedirs(network_folder_name)

split_rates = np.arange(0.1, 1.0, 0.1)

train_test_sets_splitted = get_train_test_sets_splitted(bits, split_rates, suffix=suffix)
neural_tests = 10
try_nums = 10
neural_jumps = {2: 2, 3: 2, 4: 3, 5: 4, 6: 5}[bits]
args = get_args(train_test_sets_splitted, network_folder_name, bits=bits, neural_tests=neural_tests, neural_jumps=neural_jumps, try_nums=try_nums)

print("len(args): {}".format(len(args)))

results = do_all_args(args)
results = sorted(results, key=lambda x: [x[0], x[1]])

nls, split_rates_res, misclasses_train, misclasses_test, epochs = zip(*results)

save_results_dict(nls, split_rates_res, misclasses_train, misclasses_test, epochs, suffix="_binary_adder_{}_bits{}".format(bits, suffix))

get_reshaped_array = lambda x: np.array(x).reshape((neural_tests, -1, try_nums))
get_average_list = lambda x: np.mean(x, axis=2).tolist()
get_str_list = lambda x: list(map(lambda x: map(str, x), x))

misclasses_train_array = get_reshaped_array(misclasses_train)
misclasses_test_array = get_reshaped_array(misclasses_test)
epochs_array = get_reshaped_array(epochs)

misclasses_train_avg = get_average_list(misclasses_train_array)
misclasses_test_avg = get_average_list(misclasses_test_array)
epochs_avg = get_average_list(epochs_array)

nls_str = map(lambda x: "_".join(map(str, x)), np.array(nls).astype(np.int).reshape((neural_tests, len(split_rates)*try_nums, -1))[:, 0].tolist())
split_rates_str = tuple((str(split_rate) for split_rate in split_rates))
misclasses_train_str = get_str_list(misclasses_train_avg)
misclasses_test_str = get_str_list(misclasses_test_avg)
epochs_str = get_str_list(epochs_avg)

save_results_to_csv_file(bits, nls_str, split_rates_str, misclasses_train_str, misclasses_test_str, epochs_str, suffix=suffix)

get_full_bar_plot(bits, try_nums,
                        misclasses_train_avg,
                        misclasses_test_avg,
                        epochs_avg, nls_str,
                        split_rates_str, suffix=suffix)
get_full_bar_plot(bits, try_nums,
                        np.array(misclasses_train_avg).transpose().tolist(),
                        np.array(misclasses_test_avg).transpose().tolist(),
                        np.array(epochs_avg).transpose().tolist(),
                        split_rates_str, nls_str, suffix=suffix+"_inverse")
