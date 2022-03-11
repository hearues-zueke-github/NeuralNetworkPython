#! /usr/bin/python2.7
# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import pylab
import matplotlib.ticker as tkr

import dill
import gzip
import os
import select
import sys
import time

import multiprocessing as mp
import numpy as np

from colorama import Fore, Style
from copy import deepcopy
from dotmap import DotMap
from multiprocessing import Pipe, Process

sys.path.insert(0, os.path.abspath('../utils'))
import Utils

def f_sig(X):
        return 1. / (1 + np.exp(-X))

class NeuralNetwork(Exception):
    def f_sig(self, X):
        return 1. / (1 + np.exp(-X))
    def fd_sig(self, X):
        return f_sig(X) * (1 - f_sig(X))
    def f_relu(self, x):
        return np.vectorize(lambda x: 0.2*x if x < 0. else x)(x)
    def fd_relu(self, x):
        return np.vectorize(lambda x: -0.2 if x < 0. else 1)(x)
    def f_tanh(self, x):
        return np.tanh(x)
    def fd_tanh(self, x):
        return 1 - np.tanh(x)**2

    def f_rmse(self, Y, T):
        return np.sqrt(np.mean(np.sum((Y-T)**2, axis=1)))
    def f_cecf(self, Y, T):
        return np.sum(np.sum(np.vectorize(lambda y, t: -np.log(y) if t==1. else -np.log(1-y))(Y, T), axis=1))

    def f_prediction_regression_vector(self, Y, T):
        return np.sum(np.abs(Y-T) < 0.1, axis=1)==Y.shape[1]
    def f_prediction_regression(self, Y, T):
        return np.sum(self.f_prediction_regression_vector(Y, T))
    def f_missclass_regression_vector(self, Y, T):
        return np.sum(np.abs(Y-T) < 0.1, axis=1)!=Y.shape[1]
    def f_missclass_regression(self, Y, T):
        return np.sum(self.f_missclass_regression_vector(Y, T))

    def f_prediction_vector(self, Y, T):
        return np.sum(np.vectorize(lambda x: 0 if x < 0.5 else 1)(Y).astype(np.uint8)==T.astype(np.uint8), axis=1)==Y.shape[1]
    def f_prediction(self, Y, T):
        return np.sum(self.f_prediction_vector(Y, T))
    def f_missclass_vector(self, Y, T):
        return np.sum(np.vectorize(lambda x: 0 if x < 0.5 else 1)(Y).astype(np.uint8)==T.astype(np.uint8), axis=1)!=Y.shape[1]
    def f_missclass(self, Y, T):
        return np.sum(self.f_missclass_vector(Y, T))

    def f_prediction_onehot_vector(self, Y, T):
        return np.argmax(Y, axis=1)==np.argmax(T, axis=1)
    def f_prediction_onehot(self, Y, T):
        return np.sum(np.argmax(Y, axis=1)==np.argmax(T, axis=1))
    def f_missclass_onehot_vector(self, Y, T):
        return np.argmax(Y, axis=1)!=np.argmax(T, axis=1)
    def f_missclass_onehot(self, Y, T):
        return np.sum(np.argmax(Y, axis=1)!=np.argmax(T, axis=1))

    def f_diff(self, bws, bwsd, eta):
        return map(lambda (i, a, b): a-b*i**1.5*eta, zip(np.arange(len(bws), 0, -1.), bws, bwsd))
    def f_deriv_prev_bigger(self, ps, ds):
        return np.sum(map(lambda (p, d): np.sum(np.abs(p)>np.abs(d)), zip(ps, ds)))
    def f_deriv_prev_bigger_per_layer(self, ps, ds):
        return map(lambda (p, d): np.sum(np.abs(p)>np.abs(d)), zip(ps, ds))

    def get_random_bws(self, nl):
        return np.array([np.random.uniform(-1./np.sqrt(n), 1./np.sqrt(n), (m+1, n)) for m, n in zip(nl[:-1], nl[1:])])
    
    def __init__(self, ):
        self.dpi_quality = 500

        self.bws_1 = None
        self.bws_2 = None
        self.bwsd_1 = None
        self.bwsd_2 = None

        self.prev2_diff_1 = None
        self.prev_diff_1 = None
        self.diff_1 = None

        self.bws = None
        self.bws_fixed = []
        self.nl = None
        self.nl_str = None
        self.nl_whole = []
        self.etas = []
        self.costs_train = []
        self.costs_valid = []
        self.missclasses_percent_train = []
        self.missclasses_percent_valid = []
        self.costs_missclasses_percent_test = []

        self.calc_cost = None
        self.calc_missclass = None
        self.calc_missclass_vector = None

        self.bws_best_early_stopping = 0.
        self.epoch_best_early_stopping = 0
        self.cost_valid_best_early_stopping = 10.
        self.missclass_percent_valid_best_early_stopping = 100.

        self.with_momentum_1_degree = False
        self.with_confusion_matrix = False

        self.trained_depth = 0
        self.trained_depth_prev = 0

        self.set_hidden_function("tanh")
        # self.hidden_multiplier = 1.2

        # self.f_tanh_array = np.frompyfunc(np.tanh, 1, 1)

    def set_hidden_function(self, func_str):
        if func_str == "sig":
            self.f_hidden = self.f_sig
            self.fd_hidden = self.fd_sig
        elif func_str == "relu":
            self.f_hidden = self.f_relu
            self.fd_hidden = self.fd_relu
        elif func_str == "tanh":
            self.f_hidden = self.f_tanh
            self.fd_hidden = self.fd_tanh

    def init_bws(self, X, T, hidden_layer=[]):
        self.nl = [X.shape[1]]+hidden_layer+[T.shape[1]]
        self.nl_str = "_".join(list(map(str, self.nl)))
        self.bws = self.get_random_bws(self.nl)

    def calc_feed_forward(self, X, bws):
        ones = np.ones((X.shape[0], 1))
        Y = X
        for i, bw in zip(xrange(len(bws), 0, -1), bws[:-1]):
            # Y = self.f_relu(np.hstack((ones, Y)).dot(bw))
            # Y = self.f_tanh(np.hstack((ones, Y)).dot(bw))
            Y = self.f_hidden(np.hstack((ones, Y)).dot(bw)) # *self.hidden_multiplier**i)
        Y = self.f_sig(np.hstack((ones, Y)).dot(bws[-1]))
        return Y

    def calc_feed_forward_hidden_only(self, X, bws):
        ones = np.ones((X.shape[0], 1))
        Y = X
        for bw in bws:
            Y = self.f_hidden(np.hstack((ones, Y)).dot(bw))
        return Y

    def calc_backprop(self, X, T, bws):
        Xs = []
        Ys = [X]
        ones = np.ones((X.shape[0], 1))
        Y = X
        for bw in bws[:-1]:
        # for i, bw in zip(xrange(len(bws), 0, -1), bws[:-1]):
            A = np.hstack((ones, Y)).dot(bw)#*self.hidden_multiplier**i;
            Xs.append(A)
            Y = self.f_hidden(A)
            Ys.append(Y)
        A = np.hstack((ones, Y)).dot(bws[-1]); Xs.append(A)
        Y = self.f_sig(A)
        # Y_exp = np.exp(A)
        # Y = Y_exp/np.sum(Y_exp, axis=1).reshape((Y_exp.shape[0], 1))
        Y = Y / np.sum(Y, axis=1).reshape((-1, 1))
        Ys.append(Y)

        d = (Y-T)
        bwsd = np.array([np.zeros(bwsdi.shape) for bwsdi in bws])
        bwsd[-1] = np.hstack((ones, Ys[-2])).T.dot(d)

        length = len(bws)
        for i in xrange(2, length+1):
            d = d.dot(bws[-i+1][1:].T)*self.fd_hidden(Xs[-i])#*self.hidden_multiplier**i

            # d_abs = np.abs(d)
            # max_num = np.max(d_abs)
            # min_num = np.min(d_abs)
            # if max_num < 1 and max_num > 0.000001:
            #     d /= max_num
            # if min_num > 0.0001:
            #     d *= d_abs**0.5
            # else:
            #     d *= d_abs**0.1

            # if i == length:
            bwsd[-i] = np.hstack((ones, Ys[-i-1])).T.dot(d)

        return np.array(bwsd)

    def calc_numerical_gradient(self, X, T, bws):
        bwsd = deepcopy(bws) # [np.zeros_like(bwsi) for bwsi in bws]

        epsilon = 0.00001
        for i, (bwsi, bwsdi) in enumerate(zip(bws, bwsd)):
            print("calc numerical layer nr. {}".format(i+1))
            for y in xrange(0, bwsi.shape[0]):
                print("y: {}".format(y))
                for x in xrange(0, bwsi.shape[1]):
                    print("x: {}".format(x))

                    bwsi[y, x] += epsilon
                    fr = self.calc_cost(self.calc_feed_forward(X, bws), T)
                    bwsi[y, x] -= epsilon*2.
                    fl = self.calc_cost(self.calc_feed_forward(X, bws), T)
                    bwsi[y, x] += epsilon
                    bwsdi[y, x] = (fr - fl) / (2.*epsilon)

        return bwsd

    # TODO: need a master function, which coordinates the worker_thread's
    def master_function(self):
        pass

    def worker_thread(self, pipe_in, pipe_out):
        while True:
            command, args = pipe_in.recv()

            # TODO: add a revolver chaning for the bwsd calc of train set! (e.g. calc the bwsd in 2 steps)
            if command == "exit":
                break
            elif command == "thread_nr":
                thread_nr = args[0]
            elif command == "train_set":
                X_train_full, T_train_full = args
                len_tr_full = X_train_full.shape[0]
                len_tr_samples = len_tr_full

                parts = 5
                if len_tr_full >= parts*2:
                    splits_idx = self.get_splits(len_tr_full, parts)
                    X_trains = [X_train_full[idx_start:idx_end] for idx_start, idx_end in splits_idx]
                    T_trains = [T_train_full[idx_start:idx_end] for idx_start, idx_end in splits_idx]
                else:
                    X_trains = [X_train_full]
                    T_trains = [T_train_full]

                X_train = X_train_full
                T_train = T_train_full
            elif command == "train_get_len":
                pipe_out.send((len_tr_samples, ))
            elif command == "valid_set":
                X_valid, T_valid = args
            elif command == "test_set":
                X_test, T_test = args
            elif command == "backprop":
                bws = deepcopy(args[0])
                bwsd_acc = bws*0

                # for _ in xrange(0, 3):
                #     Y_train = self.calc_feed_forward(X_train, bws)
                #     miss_class_vector = self.calc_missclass_vector(Y_train, T_train)
                #     X_train_false = X_train[miss_class_vector]
                #     T_train_false = T_train[miss_class_vector]
                #     X_train_true = X_train[miss_class_vector==0]
                #     T_train_true = T_train[miss_class_vector==0]
                #
                #     if np.sum(miss_class_vector) > 0:
                #         bwsd_acc += self.calc_backprop(X_train_false, T_train_false, bws)
                #         bwsd = bwsd_acc
                #         bws -= bwsd/X_train_false.shape[0]*0.0001
                #         bwsd_acc += self.calc_backprop(X_train_true, T_train_true, bws)
                #     else:
                #         bwsd_acc = self.calc_backprop(X_train, T_train, bws)

                bwsd_acc = self.calc_backprop(X_trains[0], T_trains[0], bws)
                bwsd = bwsd_acc

                for X_train_i, T_train_i in zip(X_trains[1:], T_trains[1:]):
                    bws -= bwsd*0.001/X_train_i.shape[0]
                    bwsd = self.calc_backprop(X_train_i, T_train_i, bws)
                    bwsd_acc += bwsd

                X_trains = X_trains[1:]+X_trains[:1]
                T_trains = T_trains[1:]+T_trains[:1]

                pipe_out.send((bwsd_acc, len_tr_samples))

                # bws = args[0]
                # pipe_out.send((self.calc_backprop(X_train, T_train, bws), len_tr_samples))
            elif command == "calc":
                bws = args[0]
                Y_train = self.calc_feed_forward(X_train, bws)
                cost_train = self.calc_cost(Y_train, T_train)
                missclass_train = self.calc_missclass(Y_train, T_train)
                
                Y_valid = self.calc_feed_forward(X_valid, bws)
                cost_valid = self.calc_cost(Y_valid, T_valid)
                missclass_valid = self.calc_missclass(Y_valid, T_valid)
                
                pipe_out.send((cost_train, missclass_train,
                               cost_valid, missclass_valid))
            elif command == "calc_valid":
                Y_valid = self.calc_feed_forward(X_valid, bws)
                cost_valid = self.calc_cost(Y_valid, T_valid)
                missclass_valid = self.calc_missclass(Y_valid, T_valid)

                pipe_out.send((cost_valid, missclass_valid))
            elif command == "calc_test":
                Y_test = self.calc_feed_forward(X_test, bws)
                cost_test = self.calc_cost(Y_test, T_test)
                missclass_test = self.calc_missclass(Y_test, T_test)

                pipe_out.send((cost_test, missclass_test))
            elif command == "calc_forward":
                bws = args[0]
                Y_train = self.calc_feed_forward(X_train, bws)
                Y_valid = self.calc_feed_forward(X_valid, bws)
                Y_test = self.calc_feed_forward(X_test, bws)
                
                pipe_out.send((Y_train, Y_valid, Y_test))
            elif command == "calc_forward_train":
                bws = args[0]
                Y_train = self.calc_feed_forward(X_train, bws)
                
                pipe_out.send((Y_train, ))

    def get_data_sets(self, data_sets, X_train, T_train, X_valid, T_valid, X_test, T_test):
        data_sets.X_train = X_train
        data_sets.T_train = T_train
        data_sets.X_valid = X_valid
        data_sets.T_valid = T_valid
        data_sets.X_test = X_test
        data_sets.T_test = T_test

    def get_splits(self, n, k):
        m = n % k
        l = n // k
        splits = np.array((np.arange(0, m)*(l+1)).tolist()+(np.arange(0, k-m+1)*l+(l+1)*m).tolist())
        splits_idx = list(zip(splits[:-1], splits[1:]))

        return splits_idx

    def get_multiprocessing_data(self, mp_data, data_sets, mapVals):
        cpu_amount = int(mp.cpu_count()*1.)
        # cpu_amount = mp.cpu_count()
        pipes_main_threads = [Pipe() for _ in xrange(0, cpu_amount)]
        pipes_threads_main = [Pipe() for _ in xrange(0, cpu_amount)]

        thread_pipes_out, main_pipes_in = list(zip(*pipes_main_threads))
        main_pipes_out, thread_pipes_in = list(zip(*pipes_threads_main))

        procs = [Process(target=self.worker_thread, args=(pipe_in, pipe_out)) for
            pipe_in, pipe_out in zip(thread_pipes_in, thread_pipes_out)]

        # set thread_nr
        for i, main_pipe_out in enumerate(main_pipes_out):
            main_pipe_out.send(("thread_nr", (i, )))
            
        def set_train_sets(length):
            splits_idx = self.get_splits(length, cpu_amount)
            print("train: splits_idx: {}".format(splits_idx))
            for main_pipe_out, (idx_start, idx_end) in zip(main_pipes_out, splits_idx):
                main_pipe_out.send(("train_set", (data_sets.X_train[idx_start:idx_end], data_sets.T_train[idx_start:idx_end])))

        def get_train_length():
            train_length = 0
            for main_pipe_out in main_pipes_out:
                main_pipe_out.send(("train_get_len", ()))
            for main_pipe_in in main_pipes_in:
                train_length += main_pipe_in.recv()[0]
            return train_length

        def set_valid_sets(length):
            splits_idx = self.get_splits(length, cpu_amount)
            for main_pipe_out, (idx_start, idx_end) in zip(main_pipes_out, splits_idx):
                main_pipe_out.send(("valid_set", (data_sets.X_valid[idx_start:idx_end], data_sets.T_valid[idx_start:idx_end])))

        def set_test_sets(length):
            splits_idx = self.get_splits(length, cpu_amount)
            for main_pipe_out, (idx_start, idx_end) in zip(main_pipes_out, splits_idx):
                main_pipe_out.send(("test_set", (data_sets.X_test[idx_start:idx_end], data_sets.T_test[idx_start:idx_end])))

        def calc_bwsd(bws):
            for main_pipe_out in main_pipes_out:
                main_pipe_out.send(("backprop", (bws, )))
            bwsd, len_tr = main_pipes_in[0].recv()
            for main_pipe_in in main_pipes_in[1:]:
                bwsd1, len_tr_samples = main_pipe_in.recv()
                len_tr += len_tr_samples
                bwsd += bwsd1

            return bwsd/float(len_tr)

        def calc_new_bws(bws, bwsd, eta):
            bws_new = bws-bwsd*eta

            return bws_new

        def calc_costs(bws):
            for main_pipe_out in main_pipes_out:
                main_pipe_out.send(("calc", (bws, )))
            sum_cost_train = 0
            sum_missclass_train = 0
            sum_cost_valid = 0
            sum_missclass_valid = 0
            for main_pipe_in in main_pipes_in:
                cost_train, missclass_train, cost_valid, missclass_valid = main_pipe_in.recv()
                sum_cost_train += cost_train
                sum_missclass_train += missclass_train
                sum_cost_valid += cost_valid
                sum_missclass_valid += missclass_valid
            cost_train = sum_cost_train/mapVals.factor_train
            missclass_percent_train = sum_missclass_train/mapVals.length_train_float*100.
            cost_valid = sum_cost_valid/mapVals.factor_valid
            missclass_percent_valid = sum_missclass_valid/mapVals.length_valid_float*100.

            return cost_train, cost_valid, missclass_percent_train, missclass_percent_valid

        def calc_cost_valid(bws):
            for main_pipe_out in main_pipes_out:
                main_pipe_out.send(("calc_valid", (bws, )))
            sum_cost_valid = 0
            sum_missclass_valid = 0
            for main_pipe_in in main_pipes_in:
                cost_valid, missclass_valid = main_pipe_in.recv()
                sum_cost_valid += cost_valid
                sum_missclass_valid += missclass_valid
            cost_valid = sum_cost_valid/mapVals.factor_valid
            missclass_percent_valid = sum_missclass_valid/mapVals.length_valid_float*100.

            return cost_valid, missclass_percent_valid

        def calc_cost_test(bws):
            for main_pipe_out in main_pipes_out:
                main_pipe_out.send(("calc_test", (bws, )))
            sum_cost_test = 0
            sum_missclass_test = 0
            for main_pipe_in in main_pipes_in:
                cost_test, missclass_test = main_pipe_in.recv()
                sum_cost_test += cost_test
                sum_missclass_test += missclass_test
            cost_test = sum_cost_test/mapVals.factor_test
            missclass_percent_test = sum_missclass_test/mapVals.length_test_float*100.

            return cost_test, missclass_percent_test

        def calc_forward(bws):
            for main_pipe_out in main_pipes_out:
                main_pipe_out.send(("calc_forward", (bws, )))
            Ys_train, Ys_valid, Ys_test = main_pipes_in[0].recv()
            for main_pipe_in in main_pipes_in[1:]:
                Ysi_train, Ysi_valid, Ysi_test = main_pipe_in.recv()
                Ys_train = np.vstack((Ys_train, Ysi_train))
                Ys_valid = np.vstack((Ys_valid, Ysi_valid))
                Ys_test = np.vstack((Ys_test, Ysi_test))

            return Ys_train, Ys_valid, Ys_test

        def calc_forward_train(bws):
            for main_pipe_out in main_pipes_out:
                main_pipe_out.send(("calc_forward_train", (bws, )))
            Ys_train = main_pipes_in[0].recv()[0]
            for main_pipe_in in main_pipes_in[1:]:
                Ysi_train = main_pipe_in.recv()[0]
                Ys_train = np.vstack((Ys_train, Ysi_train))

            return Ys_train

        mp_data.procs = procs
        mp_data.main_pipes_in = main_pipes_in
        mp_data.main_pipes_out = main_pipes_out
        mp_data.set_train_sets = set_train_sets
        # mp_data.set_next_train = set_next_train
        mp_data.get_train_length = get_train_length
        mp_data.set_valid_sets = set_valid_sets
        mp_data.set_test_sets = set_test_sets
        mp_data.calc_bwsd = calc_bwsd
        mp_data.calc_new_bws = calc_new_bws
        mp_data.calc_costs = calc_costs
        mp_data.calc_cost_valid = calc_cost_valid
        mp_data.calc_cost_test = calc_cost_test
        mp_data.calc_forward = calc_forward
        mp_data.calc_forward_train = calc_forward_train

    def get_beginning_values(self, data_sets, mp_data, mapVals):
        mapVals.bws = self.bws

        mapVals.length_train_full = data_sets.X_train.shape[0]
        mapVals.length_train_choosen = mapVals.length_train_full

        mapVals.length_valid_full = data_sets.X_valid.shape[0]
        mp_data.set_valid_sets(mapVals.length_valid_full)

        mapVals.length_test_full = data_sets.X_test.shape[0]
        mp_data.set_test_sets(mapVals.length_test_full)
        
        mapVals.factor_valid = float(data_sets.X_valid.shape[0]*data_sets.T_valid.shape[1])
        mapVals.length_valid_float = float(data_sets.X_valid.shape[0])

        mapVals.factor_test = float(data_sets.X_test.shape[0]*data_sets.T_test.shape[1])
        mapVals.length_test_float = float(data_sets.X_test.shape[0])

        mapVals.alpha = 0.1

        if len(self.costs_valid) > 3:
            # mapVals.is_full_length = self.is_full_length
            mp_data.set_train_sets(mapVals.length_train_choosen)
            mapVals.length_train = mp_data.get_train_length()
            mapVals.factor_train = float(mapVals.length_train*data_sets.T_train.shape[1])
            mapVals.length_train_float = float(mapVals.length_train)

            mapVals.bws_1 = self.bws_1
            mapVals.bws_2 = self.bws_2
            mapVals.bwsd_1 = self.bwsd_1
            mapVals.bwsd_2 = self.bwsd_2
            mapVals.eta = self.etas[-1]

            mapVals.prev2_diff_1 = self.prev2_diff_1
            mapVals.prev_diff_1 = self.prev_diff_1
            mapVals.diff_1 = self.diff_1
            mapVals.prev_cost_train = self.prev_cost_train
            mapVals.prev_cost_valid = self.prev_cost_valid
            mapVals.prev_missclass_percent_valid = self.prev_missclass_percent_valid

            mapVals.count_same_missclass_valid = self.count_same_missclass_valid
            mapVals.current_epoch = self.current_epoch
        else:
            mapVals.current_epoch = 0
            mapVals.count_same_missclass_valid = 0
            mapVals.eta = 0.0001

            # if 16 <= data_sets.X_train.shape[0]:
            #     mapVals.is_full_length = False
            #     mapVals.length_train = 16
            # else:
            # mapVals.is_full_length = True
            mp_data.set_train_sets(mapVals.length_train_choosen)
            mapVals.length_train = mp_data.get_train_length()

            mapVals.factor_train = float(mapVals.length_train*data_sets.T_train.shape[1])
            mapVals.length_train_float = float(mapVals.length_train)
            
            print("calc bwsd_2")
            mapVals.bws_2 = mapVals.bws
            print("[bw.shape for bw in mapVals.bws]: {}".format([bw.shape for bw in mapVals.bws]))
            mapVals.bwsd_2 = mp_data.calc_bwsd(mapVals.bws)
            mapVals.bws = mapVals.bws-mapVals.bwsd_2*mapVals.eta
            print("calc bwsd_2 finished")

            print("calc bwsd_1")
            mapVals.bws_1 = mapVals.bws
            mapVals.bwsd_1 = mp_data.calc_bwsd(mapVals.bws)
            mapVals.bws = mapVals.bws-mapVals.bwsd_1*mapVals.eta
            mapVals.prev_cost_train, mapVals.prev_cost_valid, \
            mapVals.prev_missclass_percent_train, mapVals.prev_missclass_percent_valid = mp_data.calc_costs(mapVals.bws)
            print("calc bwsd_1 finished")
            self.prev_cost_valid = mapVals.prev_cost_valid

            c_tst, mc_tst = mp_data.calc_cost_test(mapVals.bws)
            self.costs_missclasses_percent_test.append((0, c_tst, mc_tst))

            mapVals.prev2_diff_1 = mapVals.prev_cost_train
            mapVals.prev_diff_1 = mapVals.prev_cost_train
            mapVals.diff_1 = mapVals.prev_cost_train

            mapVals.eta = 0.1

            self.etas = [mapVals.eta]
            self.costs_train = [mapVals.prev_cost_train]
            self.costs_valid = [mapVals.prev_cost_valid]
            self.missclasses_percent_train = [mapVals.prev_missclass_percent_train]
            self.missclasses_percent_valid = [mapVals.prev_missclass_percent_valid]

    def set_beginning_values(self, mapVals, lists):
        self.count_same_missclass_valid = mapVals.count_same_missclass_valid
        self.current_epoch = mapVals.current_epoch
        self.length_train = mapVals.length_train
        # self.is_full_length = mapVals.is_full_length
        self.bws = mapVals.bws
        self.bws_1 = mapVals.bws_1
        self.bws_2 = mapVals.bws_2
        self.bwsd_1 = mapVals.bwsd_1
        self.bwsd_2 = mapVals.bwsd_2
        self.prev_cost_train = mapVals.prev_cost_train
        self.prev_cost_valid = mapVals.prev_cost_valid
        self.prev_missclass_percent_valid = mapVals.prev_missclass_percent_valid

        self.etas.extend(lists.etas)
        self.costs_train.extend(lists.costs_train)
        self.missclasses_percent_train.extend(lists.missclasses_percent_train)
        self.costs_valid.extend(lists.costs_valid)
        self.missclasses_percent_valid.extend(lists.missclasses_percent_valid)

    @staticmethod
    def do_step_by_step(network_path, rounds, iterations, file_path_data_set, bws_folder):
        file_path_network = network_path+"/whole_network.pkl.gz"

        with gzip.GzipFile(file_path_network, "rb") as f:
            nn = dill.load(f)

        data_set = np.load(file_path_data_set)
        X_train_orig = data_set["X_train"]
        T_train = data_set["T_train"]
        X_valid_orig = data_set["X_valid"]
        T_valid = data_set["T_valid"]
        X_test_orig = data_set["X_test"]
        T_test = data_set["T_test"]

        X_train = X_train_orig
        X_valid = X_valid_orig
        X_test = X_test_orig

        max_depth = 3
        if nn.trained_depth > 0 and nn.trained_depth < max_depth:
            X_train = nn.calc_feed_forward_hidden_only(X_train, nn.bws_fixed)
            X_valid = nn.calc_feed_forward_hidden_only(X_valid, nn.bws_fixed)
            X_test = nn.calc_feed_forward_hidden_only(X_test, nn.bws_fixed)

        iteration = nn.fit_network_basic_step_by_step(X_train, T_train,
                                                      X_valid, T_valid,
                                                      X_test, T_test)
        iteration.start()

        for round_i in xrange(0, rounds):
            # TODO: this condition should be changed, so that it can be applied more generell
            print("len(nn.missclasses_percent_train): {}".format(len(nn.missclasses_percent_train)))
            if len(nn.missclasses_percent_train) < 5:
                # sqr_error_tr = 1.
                diff_mc_tr = 1.
                mc_trs_sum_sqr = 1.
            else:
                mc_trs = np.array(nn.missclasses_percent_train[-50:])
                mc_trs = np.sort(mc_trs)
                mc_trs_diff = mc_trs[1:]-mc_trs[:-1]
                mc_trs_sum_sqr = np.mean(mc_trs_diff**2)
                print("mc_trs_sum_sqr: {}".format(mc_trs_sum_sqr))
                # sqr_error_tr = np.sum((mc_trs-np.mean(mc_trs))**2)
                # print("sqr_error_tr: {}".format(sqr_error_tr))
                
                diff_mc_tr = np.max(mc_trs)-np.min(mc_trs)
                print("diff_mc_tr: {}".format(diff_mc_tr))
            # if nn.missclasses_percent_train[-1] < 8. and \
            # if sqr_error_tr < 0.01 and \
            # if diff_mc_tr < 0.5 and \
            if mc_trs_sum_sqr < 0.005 and \
               len(nn.nl_whole) < max_depth:
                nn.trained_depth += 1

            if nn.trained_depth_prev != nn.trained_depth:
                if nn.trained_depth >= max_depth:
                    nn.nl_whole.append(nn.nl[0])
                    nn.nl = nn.nl_whole+nn.nl[1:]
                    nn.nl_str = "_".join(list(map(str, nn.nl)))
                    nn.bws = np.array(nn.bws_fixed+nn.bws.tolist())
                    print("Take this!")

                    iteration.finish()
                    iteration = nn.fit_network_basic_step_by_step(X_train_orig, T_train,
                                                                  X_valid_orig, T_valid,
                                                                  X_test_orig, T_test)
                    iteration.start()
                else:
                    nl_old = nn.nl
                    nl = [nl_old[1], int(nl_old[1]*0.7), nl_old[2]]
                    nn.nl_whole.append(nl_old[0])
                    nn.bws_fixed.append(nn.bws[0].copy())

                    nn.nl = nl
                    nn.nl_str = "_".join(list(map(str, nn.nl)))
                    nn.bws = Utils.create_new_bws(bws_folder, nl)

                    X_train = nn.calc_feed_forward_hidden_only(X_train, [nn.bws_fixed[-1]])
                    X_valid = nn.calc_feed_forward_hidden_only(X_valid, [nn.bws_fixed[-1]])
                    X_test = nn.calc_feed_forward_hidden_only(X_test, [nn.bws_fixed[-1]])
                    
                    iteration.finish()
                    iteration = nn.fit_network_basic_step_by_step(X_train, T_train,
                                                                  X_valid, T_valid,
                                                                  X_test, T_test)
                    iteration.start()

                iteration.mapVals.eta = 0.01
                print("calc bwsd_2")
                iteration.mapVals.bws_2 = iteration.mapVals.bws
                iteration.mapVals.bwsd_2 = iteration.mp_data.calc_bwsd(iteration.mapVals.bws)
                iteration.mapVals.bws = iteration.mapVals.bws-iteration.mapVals.bwsd_2*iteration.mapVals.eta
                print("calc bwsd_2 finished")

                print("calc bwsd_1")
                iteration.mapVals.bws_1 = iteration.mapVals.bws
                iteration.mapVals.bwsd_1 = iteration.mp_data.calc_bwsd(iteration.mapVals.bws)
                iteration.mapVals.bws = iteration.mapVals.bws-iteration.mapVals.bwsd_1*iteration.mapVals.eta
                iteration.mapVals.prev_cost_train, iteration.mapVals.prev_cost_valid, \
                iteration.mapVals.prev_missclass_percent_train, iteration.mapVals.prev_missclass_percent_valid = iteration.mp_data.calc_costs(iteration.mapVals.bws)
                print("calc bwsd_1 finished")

                nn.trained_depth_prev = nn.trained_depth
            
            iteration.next(iterations)

            nn.plot_train_test_binary_classifier(network_path, X_train, X_valid, X_test)
            if nn.with_confusion_matrix:
                iteration.get_confusion_matrix(network_path)

            print("nn.nl_whole: {}".format(nn.nl_whole))
            print("nn.nl: {}".format(nn.nl))

            if (round_i+1) % 2 == 0:
                with gzip.GzipFile(file_path_network, "wb") as f:
                    dill.dump(nn, f)

        with gzip.GzipFile(file_path_network, "wb") as f:
            dill.dump(nn, f)

        iteration.finish()

        print("Finished with all!")

    def fit_network_basic_step_by_step(self, X_train, T_train, X_valid, T_valid, X_test, T_test, with_multiprocessing=False):
        class Iteration(Exception):
            def __init__(other):
                other.data_sets = DotMap()
                other.mp_data = DotMap()
                other.mapVals = DotMap()

            def start(other):
                self.get_data_sets(other.data_sets, X_train, T_train, X_valid, T_valid, X_test, T_test)
                self.get_multiprocessing_data(other.mp_data, other.data_sets, other.mapVals)

                for proc in other.mp_data.procs: proc.start()
                # TODO: make a master process
                # TODO: split train, valid, test equaly to every process
                # TODO: revolver spin the the train sets
                #       (everytime when e.g. 40 train samples
                #        are selected)
                # TODO: if all samples are choosen, then no rotation needed
                #       (train, valid)
                self.get_beginning_values(other.data_sets, other.mp_data, other.mapVals)

            def next(other, epochs=10):
                # data_sets = other.data_sets
                mp_data = other.mp_data
                mapVals = other.mapVals

                start_time = time.time()
                
                old_current_epoch = mapVals.current_epoch
                # alpha = 0.1
                alphas = [0.0]
                if self.with_momentum_1_degree:
                    alphas = [0.1]
                    # alphas = [0.1, 0.5]
                    # alphas += [0.03, 0.2] # , 0.4]

                lists = DotMap()
                lists.etas = [] 
                lists.costs_train = []
                lists.missclasses_percent_train = []
                lists.costs_valid = []
                lists.missclasses_percent_valid = []

                eta_mult_1 = 0.6**np.arange(1, 0, -1)
                eta_mult_2 = 1.1**np.arange(1, 2, 1)
                eta_multipliers = np.array(eta_mult_1.tolist()+[1]+eta_mult_2.tolist())
                eta_multipliers += (np.random.random(eta_multipliers.shape)-0.5)*0.1

                eta_max = 5.
                eta_min = 0.0001

                eta_now = mapVals.eta
                count_same_missclass_test = 0
                # TODO: eliminate all not correct calculated samples for easier learning!
                for epoch in xrange(1, epochs+1):
                    mapVals.current_epoch += 1

                    etas = eta_now*eta_multipliers
                    bws = mapVals.bws
                    bws_diff = bws-mapVals.bws_1
                    alpha_bwsd_a = [(alpha, mp_data.calc_bwsd(bws+alpha*bws_diff)) for alpha in alphas]

                    arr_eta_alpha_bwsd_a = [(e, a, bwsd_a) for e in etas for a, bwsd_a in alpha_bwsd_a]

                    all_errors = []
                    for eta, _, bwsd_a in arr_eta_alpha_bwsd_a:
                        bws_new_temp = mp_data.calc_new_bws(bws, bwsd_a, eta)
                        c_tr_temp, c_vld_temp, mc_tr_temp, mc_vld_temp = mp_data.calc_costs(bws_new_temp)
                        # c_vld_temp, mc_vld_temp = mp_data.calc_cost_valid(bws_new_temp)

                        all_errors.append(c_tr_temp)
                        # all_errors.append(c_vld_temp)

                    eta_now, alpha_now, bwsd = arr_eta_alpha_bwsd_a[np.argmin(all_errors)]

                    if eta_now > eta_max:
                        eta_now = eta_max
                    elif eta_now < eta_min:
                        eta_now = eta_min

                    bws_alpha = bws+alpha_now*(bws-mapVals.bws_1)
                    bws_new = mp_data.calc_new_bws(bws_alpha, bwsd, eta_now)

                    # c_vld, mc_vld = mp_data.calc_cost_valid(bws_new)
                    c_tr, c_vld, mc_tr, mc_vld = mp_data.calc_costs(bws_new)

                    if self.cost_valid_best_early_stopping > c_vld:
                        self.bws_best_early_stopping = deepcopy(bws_new)
                        self.epoch_best_early_stopping = mapVals.current_epoch
                        self.cost_valid_best_early_stopping = c_vld
                        self.missclass_percent_valid_best_early_stopping = mc_vld

                    lists.etas.append(eta_now)
                    lists.costs_train.append(c_tr)
                    lists.missclasses_percent_train.append(mc_tr)
                    lists.costs_valid.append(c_vld)
                    lists.missclasses_percent_valid.append(mc_vld)

                    # print("epoch: {:5}, eta: {:9.7f}, alpha: {:6.4f}, cecf_vld: {:>9.7f}, miss_vld: {:>6.3f}%, len_tr: {}".format(
                    #     mapVals.current_epoch, eta_now, alpha_now, c_vld, mc_vld, mapVals.length_train))
                    output_line = ""
                    output_line += "epoch: {:5}, eta: {:08.6f}, alpha: {:06.4f}, len_tr: {}, nl: {}".format(
                        mapVals.current_epoch, eta_now, alpha_now, mapVals.length_train, self.nl)
                    output_line += "\n"+"tr:  "+ \
                                   "cecf: "+Fore.CYAN+" {:>8.06f}".format(c_tr)+Style.RESET_ALL+", "+ \
                                   "mc: "+Fore.CYAN+"{:>5.2f}".format(mc_tr)+Style.RESET_ALL+"%"
                    output_line += "\n"+"vld: "+ \
                                   "cecf: "+Fore.YELLOW+" {:>8.06f}".format(c_vld)+Style.RESET_ALL+", "+ \
                                   "mc: "+Fore.YELLOW+"{:>5.2f}".format(mc_vld)+Style.RESET_ALL+"%"
                    print(output_line)

                    # output_line = ""
                    # output_line += "epoch: {:5}, eta: {:08.6f}, alpha: {:06.4f}, len_tr: {}".format(
                    #     mapVals.current_epoch, eta_now, alpha_now, mapVals.length_train)
                    # output_line += "\n"+"cecf: "+ \
                    #                "tr: "+Fore.CYAN+" {:>8.06f}".format(c_tr)+Style.RESET_ALL+", "+ \
                    #                "vld: "+Fore.YELLOW+" {:>8.06f}".format(c_vld)+Style.RESET_ALL
                    # output_line += "\n"+"miss: "+ \
                    #                "tr: "+Fore.CYAN+"{:>05.2f}".format(mc_tr)+Style.RESET_ALL+",     "+ \
                    #                "vld: "+Fore.YELLOW+"{:>05.2f}".format(mc_vld)+Style.RESET_ALL
                    # print(output_line)

                    # print("epoch: {:5}, eta: {:8.6f}, alpha: {:6.4f}, cecf_tr: {:>8.6f}, miss_tr: {:>5.2f}%, cecf_vld: {:>8.6f}, miss_vld: {:>5.2f}%, len_tr: {}".format(
                    #     mapVals.current_epoch, eta_now, alpha_now, c_tr, mc_tr, c_vld, mc_vld, mapVals.length_train))
                    taken_time = time.time()-start_time
                    print(Fore.GREEN+"Taken time: {:.5}s, ~{:.5f}s/epoch".format(taken_time, taken_time/epoch)+Style.RESET_ALL)

                    mapVals.bws_2 = mapVals.bws_1
                    mapVals.bws_1 = mapVals.bws
                    mapVals.bws = bws_new

                    mapVals.bwsd_2 = mapVals.bwsd_1
                    mapVals.bwsd_1 = mapVals.bwsd
                    mapVals.bwsd = bwsd

                    mapVals.prev_cost_train = c_tr
                    mapVals.prev_cost_valid = c_vld
                    mapVals.prev_missclass_percent_valid = mc_vld
                    mapVals.eta = eta_now

                    # TODO: need a break condition!
                    # mapVals.bws = bws_new

                    # if mc_tr == 0.:
                    #     count_same_missclass_test += 1
                    # else:
                    #     count_same_missclass_test = 0
                    #
                    # if mc_tr == 0. and count_same_missclass_test > 100:
                    #     break

                self.set_beginning_values(mapVals, lists)
                
                c_tst, mc_tst = mp_data.calc_cost_test(self.bws_best_early_stopping)
                self.costs_missclasses_percent_test.append((self.current_epoch, c_tst, mc_tst))
                print(Fore.LIGHTMAGENTA_EX+"    cecf_tst: {:>8.6f}, miss_tst: {:>5.2f}%".format(
                        c_tst, mc_tst)+Style.RESET_ALL)

                taken_time = time.time()-start_time
                print(Fore.GREEN+"Taken time: {:.5}s, ~{:.5f}s/epoch".format(taken_time, taken_time/epoch)+Style.RESET_ALL)

            def get_confusion_matrix(other, network_path):
                data_sets = other.data_sets
                mp_data = other.mp_data
                mapVals = other.mapVals

                Y_train, Y_valid, Y_test = mp_data.calc_forward(mapVals.bws)

                Yn_train = np.argmax(Y_train, axis=1)
                Tn_train = np.argmax(data_sets.T_train, axis=1)
                Yn_valid = np.argmax(Y_valid, axis=1)
                Tn_valid = np.argmax(data_sets.T_valid, axis=1)
                Yn_test = np.argmax(Y_test, axis=1)
                Tn_test = np.argmax(data_sets.T_test, axis=1)

                self.plot_confusion_matrix(network_path,
                                           "confusion_matrix", Yn_train, Tn_train,
                                                               Yn_valid, Tn_valid,
                                                               Yn_test, Tn_test, classes=10)

            def finish(other):
                for main_pipe_out in other.mp_data.main_pipes_out: main_pipe_out.send(("exit", ((), )))
                for proc in other.mp_data.procs: proc.join()

        return Iteration()

    def plot_train_test_binadder(self, folder_path, bits, operation):
        plots_file_name_prefix = folder_path+"/binary_adder_bits_{}_nl_{}".format(bits, self.nl_str)
        self.plot_train_valid_test_graphs(plots_file_name_prefix, title_inffix="{} bits {}, ".format(bits, operation))

    def plot_train_test_binary_classifier(self, network_folder, X_train, X_valid, X_test):
        plots_file_name_prefix = "plot_train_{}_valid_{}_test_{}".format(
            X_train.shape[0], X_valid.shape[0], X_test.shape[0])
        # plots_file_name_prefix = "plot_valid_{}_test_{}".format(
        #     X_valid.shape[0], X_test.shape[0])
        self.plot_train_valid_test_graphs(network_folder, plots_file_name_prefix)

    def plot_train_valid_test_graphs(self, network_folder, plots_file_name_prefix, title_inffix=""):
        def plot_cecfs():
            # CECFs plots
            fig, ax = plt.subplots(figsize=(10, 5), num=101)

            plt.title("Cross Entropy Cost Function,{} Network Layers: {}".format(title_inffix, self.nl)+
                      "\nwith_momentum_1_degree={}".format(self.with_momentum_1_degree))
            plt.xlabel("epoch")
            plt.ylabel("cecf")

            x_test, costs_test, missclasses_test = list(zip(*self.costs_missclasses_percent_test))

            x = np.arange(0, len(self.costs_valid))
            # x = np.arange(0, len(self.costs_train))
            cost_train_plot = plt.plot(x, self.costs_train, "b-", linewidth=1)[0]
            cost_valid_plot = plt.plot(x, self.costs_valid, "g-", linewidth=1)[0]
            cost_test_plot = plt.plot(x_test, costs_test, "r-", linewidth=1)[0]

            ax.set_yscale("log", nonposy='clip')

            # plt.legend((cost_valid_plot, cost_test_plot),
            #            ("valid", "test"), loc="upper right")
            plt.legend((cost_train_plot, cost_valid_plot, cost_test_plot),
                       ("train", "valid", "test"), loc="upper right")
            plt.savefig(network_folder+"/"+plots_file_name_prefix+"_costs.png", format="png", dpi=self.dpi_quality)

        def plot_missclasses():
            # Missclass rate plots
            fig, ax = plt.subplots(figsize=(12, 7), num=102)

            plt.title("Missclassification rate plot,{} Network Layers: {}".format(title_inffix, self.nl)+
                      "\nwith_momentum_1_degree={}".format(self.with_momentum_1_degree))
            plt.xlabel("epoch")
            plt.ylabel("missclass rate in %")
            plt.ylim([-0.1, 100.1])

            x_test, costs_test, missclasses_test = list(zip(*self.costs_missclasses_percent_test))

            x = np.arange(0, len(self.missclasses_percent_valid))
            # x = np.arange(0, len(self.missclasses_percent_train))
            missclasses_train_plot = plt.plot(x, self.missclasses_percent_train, "b-", linewidth=1)[0]
            missclasses_valid_plot = plt.plot(x, self.missclasses_percent_valid, "g-", linewidth=1)[0]
            missclasses_test_plot = plt.plot(x_test, missclasses_test, "r-", linewidth=1)[0]

            # plt.legend((missclasses_valid_plot, missclasses_test_plot),
            #            ("valid", "test"), loc="upper right")
            plt.legend((missclasses_train_plot, missclasses_valid_plot, missclasses_test_plot),
                       ("train", "valid", "test"), loc="upper right")
            plt.savefig(network_folder+"/"+plots_file_name_prefix+"_missclass.png", format="png", dpi=self.dpi_quality)

        def plot_eta():
            # Eta rate plots
            fig, ax = plt.subplots(figsize=(12, 7), num=103)

            plt.title("Eta rate plot,{} Network Layers: {}".format(title_inffix, self.nl)+
                      "\nwith_momentum_1_degree={}".format(self.with_momentum_1_degree))
            plt.xlabel("epoch")
            plt.ylabel("eta")
            ax.set_yscale("log", nonposy='clip')

            x = np.arange(0, len(self.etas))
            eta_plot = plt.plot(x, self.etas, "b.", linewidth=3)[0]

            plt.legend((eta_plot, ), ("eta", ), loc="upper right")
            plt.savefig(network_folder+"/"+plots_file_name_prefix+"_eta.png", format="png", dpi=self.dpi_quality)

        funcs = [plot_cecfs, plot_missclasses, plot_eta]
        procs = [mp.Process(target=func) for func in funcs]

        for proc in procs: proc.start()
        for proc in procs: proc.join()

        plt.close("all")

    def plot_confusion_matrix(self, network_path,  plots_file_name_prefix,
            Yn_tr, Tn_tr, Yn_vld, Tn_vld, Yn_tst, Tn_tst, classes=10):
        # help from: http://stackoverflow.com/questions/2897826/confusion-matrix-with-number-of-classified-misclassified-instances-on-it-python
        conf_tr = np.zeros((classes, classes)).astype(np.int)
        conf_vld = np.zeros((classes, classes)).astype(np.int)
        conf_tst = np.zeros((classes, classes)).astype(np.int)

        for y, t in zip(Yn_tr, Tn_tr): conf_tr[y, t] += 1
        for y, t in zip(Yn_vld, Tn_vld): conf_vld[y, t] += 1
        for y, t in zip(Yn_tst, Tn_tst): conf_tst[y, t] += 1

        conf_tr_norm = conf_tr / (np.sum(1.*conf_tr, axis=1)).reshape((-1, 1))
        conf_vld_norm = conf_vld / (np.sum(1.*conf_vld, axis=1)).reshape((-1, 1))
        conf_tst_norm = conf_tst / (np.sum(1.*conf_tst, axis=1)).reshape((-1, 1))

        x = -0.4
        y = 0.2

        def get_confusion_plot(conf_matrix, conf_matrix_norm, inffix_str, suffix, num):
            fig, ax = plt.subplots(figsize=(12, 7), num=num)
            res = ax.imshow(pylab.array(conf_matrix_norm), cmap=plt.cm.jet, interpolation='nearest')
            for i, conf in enumerate(conf_matrix):
                for j, c in enumerate(conf):
                    if c > 0:
                        plt.text(j+x, i+y, c, fontsize=11)
            plt.title("Confusion matrix of {} data".format(inffix_str)+
                      "\nwith_momentum_1_degree={}".format(self.with_momentum_1_degree), y=1.08)
            plt.xlabel("Real digit")
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position("top")
            plt.ylabel("Predicted digit")
            plt.xticks(np.arange(0, classes))
            plt.yticks(np.arange(0, classes))
            plt.subplots_adjust(bottom=0.1, top=0.85)#left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            cb = fig.colorbar(res)
            plt.savefig(network_path+"/"+plots_file_name_prefix+"_confusion_matrix_{}.png".format(suffix), format="png")
            plt.close()

        args = [(conf_tr, conf_tr_norm, "Train", "train", 104),
                (conf_vld, conf_vld_norm, "Valid", "valid", 105),
                (conf_tst, conf_tst_norm, "Test", "test", 106)]
        procs = [mp.Process(target=get_confusion_plot, args=arg) for arg in args]

        for proc in procs: proc.start()
        for proc in procs: proc.join()

        plt.close("all")

    def gradient_check(self, X, T):
        bws = self.bws

        bwsd_real = self.calc_backprop(X, T, bws)
        bwsd_numerical = self.calc_numerical_gradient(X, T, bws)

        print("X:\n{}".format(X))
        print("T:\n{}".format(T))
        for i, (bwsdi, bwsdi_num) in enumerate(zip(bwsd_real, bwsd_numerical)):
            print("i: {}, bwsdi:\n{}".format(i, bwsdi))
            print("i: {}, bwsdi_num:\n{}".format(i, bwsdi_num))
            # print("i: {}, bwsdi > 0 and bwsdi_num > 0:\n{}".format(i, np.logical_or(np.logical_and(bwsdi>0, bwsdi_num>0), np.logical_and(bwsdi<=0, bwsdi_num<=0))))
            # print("i: {}, bwsdi_num/bwsdi:\n{}".format(i, bwsdi_num/bwsdi))

        sys.exit(0)
