#! /usr/bin/python2.7

import matplotlib
matplotlib.use('AGG')

import dill
import gzip
import os
import shutil
import socket
import sys
import unittest

import multiprocessing as mp
import numpy as np

import matplotlib.pyplot as plt

from colorama import Fore, Style
from copy import deepcopy
from dotmap import DotMap
from multiprocessing import Pipe, Process

class NeuralNetwork(Exception):
    def __init__(self, alpha=0.1, eta=0.1):
        self.bws = None
        self.bws_prev = None
        self.bwsd_prev = None

        self.iterations = 10
        self.alpha = alpha
        self.eta = eta
        self.errors_train = []
        self.errors_valid = []
        self.errors_test = []
        self.etas = []

        self.eta_max = 10.
        self.eta_min = 0.00001

        eta_mult1 = 0.7**np.arange(6, -1, -1)
        eta_mult2 = 1.1**np.arange(1, 6)
        self.eta_multipliers = np.array(eta_mult1.tolist()+eta_mult1.tolist())
        self.eta_multipliers += (np.random.random(self.eta_multipliers.shape)-0.5)*0.02

        self.f_sig = lambda X: 1/(1+np.exp(-X))
        self.fd_sig = lambda X: self.f_sig(X)*(1-self.f_sig(X))

        self.f_tanh = lambda X: np.tanh(X)
        self.fd_tanh = lambda X: 1 - np.tanh(X)**2

        f_relu_vect = np.vectorize(lambda X: 0. if X < 0. else X)
        fd_relu_vect = np.vectorize(lambda X: 0. if X < 0. else 1)

        self.f_relu = lambda X: f_relu_vect(X)
        self.fd_relu = lambda X: fd_relu_vect(X)

        self.f_hidden_func_str = "tanh"
        self.f_hidden = self.f_tanh
        self.fd_hidden = self.fd_tanh

    def set_hidden_function(self, func_str):
        self.f_hidden_func_str = func_str
        
        if func_str == "sig":
            self.f_hidden = self.f_sig
            self.fd_hidden = self.fd_sig
        elif func_str == "tanh":
            self.f_hidden = self.f_tanh
            self.fd_hidden = self.fd_tanh
        elif func_str == "relu":
            self.f_hidden = self.f_relu
            self.fd_hidden = self.fd_relu

    def err_mse(self, X, bws, T):
        return np.sum((self.calc_forward(X, bws)-T)**2)/2/T.shape[0]/T.shape[1]

    def calc_forward(self, X, bws):
        ones = np.ones((X.shape[0], 1))
        Y = X
        for bw in bws[:-1]:
            Y = self.f_hidden(np.hstack((ones, Y)).dot(bw))
        Y = self.f_sig(np.hstack((ones, Y)).dot(bws[-1]))

        return Y

    def calc_backprop(self, X, bws, T):
        Xs = []
        Ys = [X]
        ones = np.ones((X.shape[0], 1))
        Y = X
        for i, bw in zip(xrange(len(bws), 0, -1), bws[:-1]):
            A = np.hstack((ones, Y)).dot(bw); Xs.append(A)
            Y = self.f_hidden(A); Ys.append(Y)
        A = np.hstack((ones, Y)).dot(bws[-1]); Xs.append(A)
        Y = self.f_sig(A); Ys.append(Y)

        d = self.fd_sig(Xs[-1])*(Y-T)
        bwsd = deepcopy(bws)
        bwsd[-1] = np.hstack((ones, Ys[-2])).T.dot(d)
        
        for i in xrange(2, len(bws)+1):
            d = d.dot(bws[-i+1][1:].T)*self.fd_hidden(Xs[-i])
            d_abs = np.abs(d)
            max_num = np.max(d_abs)
            min_num = np.min(d_abs)
            if max_num < 1 and max_num > 0.000001:
                d /= max_num
            if min_num < 0.0001:
                d *= np.sqrt(d_abs)

            bwsd[-i] = np.hstack((ones, Ys[-i-1])).T.dot(d)

        return bwsd

    def get_trained_network(self, data_values):
        X_train = data_values.X_train
        T_train = data_values.T_train
        X_valid = data_values.X_valid
        T_valid = data_values.T_valid
        X_test = data_values.X_test
        T_test = data_values.T_test

        eta_now = self.eta
        eta_changed_min = eta_now
        eta_changed_max = eta_now
        errors_train = []
        errors_valid = []
        errors_test = []
        etas = []
        alpha = self.alpha
        error_prev = self.err_mse(X_train, self.bws, T_train)
        for i in xrange(0, self.iterations):
            bws_1 = self.bws+alpha*(self.bws-self.bws_prev)
            bwsd = self.calc_backprop(X_train, bws_1, T_train)
            # etas = np.arange(0.001, 4/np.sqrt(i+1), 0.2/np.sqrt(i+1)) # also works

            etas_now = eta_now*self.eta_multipliers
            # error_first = err_mse(X_train, self.bws, T_train)
            errors_now = np.array([self.err_mse(X_train, self.bws-eta*bwsd, T_train) for eta in etas_now])

            min_idx = np.argmin(errors_now)
            # print("min_idx: {}".format(min_idx))
            eta_now = etas_now[min_idx]
            error_train = errors_now[min_idx]
            
            if eta_now < self.eta_min:
                eta_now = self.eta_min
            if eta_now > self.eta_max:
                eta_now = self.eta_max

            if eta_changed_min > eta_now:
                eta_changed_min = eta_now
            if eta_changed_max < eta_now:
                eta_changed_max = eta_now

            # print("i: {}, eta_now: {:>10.7f}, , error_train: {:>10.7f}".format(i, eta_now, error_train))
            etas.append(eta_now)
            errors_train.append(error_train)
            
            # if error_prev > error_train and error_prev-error_train < 10**-6:
            #     break
            error_prev = error_train

            self.bwsd_prev = bwsd
            self.bws_prev = self.bws
            self.bws = self.bws-eta_now*bwsd

            error_valid = self.err_mse(X_valid, self.bws, T_valid)
            error_test = self.err_mse(X_test, self.bws, T_test)

            errors_valid.append(error_valid)
            errors_test.append(error_test)

        # print("eta_changed_min: {}".format(eta_changed_min))
        # print("eta_changed_max: {}".format(eta_changed_max))

        self.eta = eta_now

        self.errors_train.extend(errors_train)
        self.errors_valid.extend(errors_valid)
        self.errors_test.extend(errors_test)
        self.etas.extend(etas)

def get_random_bws(nl):
    l = [np.random.uniform(-1./np.sqrt(n), 1./np.sqrt(n), (m+1, n)) for m, n in zip(nl[:-1], nl[1:])]
    larray = np.empty(len(l), dtype=object)
    larray[:] = l
    return larray

get_random_matrix = lambda shape: np.random.random(shape)*2-1

def split_list(l, chunks): # get chunks of the list l
    assert isinstance(l, list)
    length = len(l)
    assert 0 < length
    assert 0 < chunks <= length

    n = length//chunks
    n1 = n+1
    len1 = (length%chunks)*n1
    return [l[i:i+n1] for i in xrange(0, len1, n1)]+\
           [l[i:i+n] for i in xrange(len1, length, n)]

# class TestCases(unittest.TestCase):
#     def testOne(self):
#         l = [1,2,3,4,5,6,7,8,9,10]
#         l3 = [[1,2,3,4], [5,6,7], [8,9,10]]
#         self.failUnless([]==split_list(l, 3))

# unittest.main()

def get_plots(nns, title, file_path_name):
    fig, arr = plt.subplots(2, len(nns), figsize=(len(nns)*4, 11))
    plt.suptitle(title, fontsize=16)
    # arr[0, 0].set_title("Best Errors")
    # arr[1, 0].set_title("Best Etas")
    
    all_errors_min = []
    all_errors_max = []
    all_etas_min = []
    all_etas_max = []
    for nn in nns:
        arrays = (nn.errors_train, nn.errors_valid, nn.errors_test)
        all_errors_min.append(np.min(arrays))
        all_errors_max.append(np.max(arrays))
        all_etas_min.append(np.min(nn.etas))
        all_etas_max.append(np.max(nn.etas))

    error_min = np.min(all_errors_min)
    error_max = np.max(all_errors_max)

    eta_min = np.min(all_etas_min)
    eta_max = np.max(all_etas_max)

    for i, nn in enumerate(nns):
        alpha_str = "{:4.2f}".format(nn.alpha)
        nl_str = "_".join(list(map(str, nn.nl)))

        arr[0, i].set_title("Errors, alpha: {}\nnl: {}".format(alpha_str, nl_str))
        arr[0, i].set_ylim(error_min, error_max)
        arr[0, i].set_yscale("log", nonposy='clip')

        p_train = arr[0, i].plot(nn.errors_train, "b-")[0]
        p_train.set_label("train")
        p_valid = arr[0, i].plot(nn.errors_valid, "g-")[0]
        p_valid.set_label("valid")
        p_test = arr[0, i].plot(nn.errors_test, "r-")[0]
        p_test.set_label("test")
        arr[0, i].legend()
    
        arr[1, i].set_title("Eta, alpha: {}\nnl: {}".format(alpha_str, nl_str))
        # arr[1, i].set_title("Eta, alpha: {}".format(alpha_str))
        arr[1, i].set_ylim(eta_min, eta_max)
        arr[1, i].set_yscale("log", nonposy='clip')

        p_eta = arr[1, i].plot(nn.etas, "b.")[0]
        p_eta.set_label("eta")
        arr[1, i].legend()

    plt.subplots_adjust(left=0.03, bottom=0.05, right=0.98, top=0.90, wspace=0.1, hspace=0.18)
    
    plt.savefig(file_path_name)
    # plt.show()

def worker_thread(pipe_in, pipe_out):
    proc_nr, nn_file_paths, data_values = pipe_in.recv()

    print("start proc_nr #{}".format(proc_nr))
    for i, nn_file_path in enumerate(nn_file_paths):
        with gzip.GzipFile(nn_file_path, "rb") as f:
            nn = dill.load(f)

        print("proc_nr #{}: nn #{} with {} hid func".format(proc_nr, i, nn.f_hidden_func_str))

        nn.get_trained_network(data_values)

        with gzip.GzipFile(nn_file_path, "wb") as f:
            dill.dump(nn, f)

    print("finish proc_nr #{}".format(proc_nr))

def get_data_set(file_path, m, k, n):
    if not os.path.exists(file_path):
        X_train = get_random_matrix((m, k))
        X_valid = get_random_matrix((int(m*0.6), k))
        X_test = get_random_matrix((int(m*0.6), k))
        
        M = get_random_matrix((k, n))
        T_train = X_train.dot(M)
        T_valid = X_valid.dot(M)
        T_test = X_test.dot(M)

        T_train[T_train>=0.] = 1.
        T_train[T_train<0.] = 0.
        T_valid[T_valid>=0.] = 1.
        T_valid[T_valid<0.] = 0.
        T_test[T_test>=0.] = 1.
        T_test[T_test<0.] = 0.

        data_values = DotMap()
        data_values.X_train = X_train
        data_values.T_train = T_train
        data_values.X_valid = X_valid
        data_values.T_valid = T_valid
        data_values.X_test = X_test
        data_values.T_test = T_test

        with gzip.GzipFile(file_path, "wb") as f:
            dill.dump(data_values, f)
    else:
        with gzip.GzipFile(file_path, "rb") as f:
            data_values = dill.load(f)

    return data_values

def test_different_architectures(network_nr):
    hostname = socket.gethostname()
    if "figi" in hostname:
        full_path_folder = "networks_gradient_descent"
    else:
        home = os.path.expanduser("~")
        full_path_folder = home+"/Documents/networks_gradient_descent"

    # file_path_accumulate_data = full_path_folder+"/accumulate_data.pkl.gz"

    # if not os.path.exists(file_path_accumulate_data):
    #     accumulate_data = DotMap()
    #     accumulate_data.network_nr = 0

    #     with gzip.GzipFile(file_path_accumulate_data, "wb") as f:
    #         dill.dump(accumulate_data, f)
    # else:
    #     with gzip.GzipFile(file_path_accumulate_data, "rb") as f:
    #         accumulate_data = dill.load(f)

    # accumulate_data.network_nr += 1
    # network_nr = accumulate_data.network_nr
    full_path_networks_folder = full_path_folder+"/try_nr_{}".format(network_nr)

    path_network = full_path_networks_folder+"/networks"
    path_pictures = full_path_networks_folder+"/pictures"
    path_data = full_path_folder+"/data"
    
    # if not os.path.exists(path_network):
    #     os.makedirs(path_network)
    if not os.path.exists(path_pictures):
        os.makedirs(path_pictures)
    if not os.path.exists(path_data):
        os.makedirs(path_data)

    m = 500
    k = 10
    n = 3
    file_path_data = path_data+"/data_set_m_{}_k_{}_n_{}.pkl.gz".format(m, k, n)
    data_values = get_data_set(file_path_data, m, k, n)
    
    nl_hiddens = [[20, 20], [20, 20, 20], [30, 20, 20], [30, 20, 10]]

    for nl_nr, nl_hidden in enumerate(nl_hiddens):
        path_network_nl_nr = path_network+"/nn_nl_nr_{}".format(nl_nr)
        if not os.path.exists(path_network_nl_nr):
            os.makedirs(path_network_nl_nr)
        nl = [k]+nl_hidden+[n]
        # nl = [k, 30, 20, 20, n]
        nl_str = "_".join(list(map(str, nl)))

        eta_start = 0.05
        cpu_amount = mp.cpu_count()

        bws = get_random_bws(nl)
        nn = NeuralNetwork()
        
        bws_prev = bws
        bwsd_prev = nn.calc_backprop(data_values.X_train, bws, data_values.T_train)
        bws = bws-eta_start*bwsd_prev

        alphas = np.arange(0, 8)*0.1
        # alphas = np.arange(0, cpu_amount)*0.1
        alphas_str = list(map(lambda x: "{:4.2f}".format(x), alphas))
        alphas_str_under = list(map(lambda x: x.replace(".", "_"), alphas_str))
        
        iterations = 800
        func_strs = ["sig", "tanh", "relu"]
        nn_file_paths_all = []
        nn_file_paths_combined = []
        for func_str in func_strs:
            # print("")
            nns = [NeuralNetwork(eta=eta_start) for _ in xrange(0, len(alphas))]
            
            for nn, alpha in zip(nns, alphas):
                nn.nl = deepcopy(nl)
                nn.bws = deepcopy(bws)
                nn.bws_prev = deepcopy(bws_prev)
                nn.alpha = alpha
                nn.iterations = iterations
                nn.set_hidden_function(func_str)

            nn_file_paths = [path_network_nl_nr+"/network_alpha_{}_func_{}.pkl.gz".format(alpha_str_under, func_str)
                                 for alpha_str_under in alphas_str_under]
            for nn, nn_file_path in zip(nns, nn_file_paths):
                with gzip.GzipFile(nn_file_path, "wb") as f:
                    dill.dump(nn, f)

            nn_file_paths_all.extend(nn_file_paths)
            nn_file_paths_combined.append((func_str, nn_file_paths))

        nn_file_paths_mixed = np.array(nn_file_paths_all)[
                              np.random.permutation(np.arange(0, len(nn_file_paths_all)))].tolist()

        nn_file_paths_chunks = split_list(nn_file_paths_mixed, cpu_amount)

        pipes_main_threads = [Pipe() for _ in xrange(0, cpu_amount)]
        pipes_threads_main = [Pipe() for _ in xrange(0, cpu_amount)]

        thread_pipes_out, main_pipes_in = list(zip(*pipes_main_threads))
        main_pipes_out, thread_pipes_in = list(zip(*pipes_threads_main))

        procs = [Process(target=worker_thread, args=(pipe_in, pipe_out)) for
                pipe_in, pipe_out in zip(thread_pipes_in, thread_pipes_out)]

        for proc in procs:
            proc.start()

        for i, (main_pipe_out, nn_file_paths) in enumerate(zip(main_pipes_out, nn_file_paths_chunks)):
            main_pipe_out.send((i, nn_file_paths, data_values))

        for proc in procs:
            proc.join()

        print("")
        title_template = "With {} hidden activation function"
        file_paths_picture = []
        for func_str, nn_file_paths in nn_file_paths_combined:
            nns = []
            for nn_file_path in nn_file_paths:
                with gzip.GzipFile(nn_file_path, "rb") as f:
                    nn = dill.load(f)
                nns.append(nn)

            file_path_picture = path_pictures+"/nl_nr_{}_func_{}_alphas.png".format(nl_nr, func_str)
            get_plots(nns, title_template.format(func_str), file_path_picture)

            print("finish plot with {} function".format(func_str))

        shutil.rmtree(path_network_nl_nr)

    shutil.rmtree(path_network)

    # with gzip.GzipFile(file_path_accumulate_data, "wb") as f:
    #     dill.dump(accumulate_data, f)

if __name__ == "__main__":
    argv = sys.argv
    assert len(argv) > 1
    network_nr_str = argv[1]
    # TODO: add check if possible to cast to int
    # assert int(network_nr_str)
    network_nr = int(network_nr_str)
    test_different_architectures(network_nr)

    os.execvp("htop", ("htop", ))
