#! /usr/bin/python2.7
# -*- coding: utf-8 -*-

import colorama
import cProfile
import csv
import dill
import gzip
import multiprocessing as mp
import os
import sys
import time

import cPickle as pkl
import numpy as np

from copy import deepcopy
from multiprocessing import Pipe, Process
from PIL import Image

from neuralnetwork import NeuralNetwork

sys.path.insert(0, os.path.abspath('../utils'))
import HashFunction as hfunc
import Utils
import UtilsBinary

cpu_amount = mp.cpu_count()
pipes_main_threads = [Pipe() for _ in xrange(0, cpu_amount)]
pipes_threads_main = [Pipe() for _ in xrange(0, cpu_amount)]

thread_pipes_out, main_pipes_in = list(zip(*pipes_main_threads))
main_pipes_out, thread_pipes_in = list(zip(*pipes_threads_main))

def get_splits(length, k):
    a = np.arange(0, length)
    m = length // k
    m1 = m+1
    c = length % k
    next_idx = m1*c
    splits = [np.arange(m1*i, m1*(i+1)) for i in xrange(0, c)]+[next_idx+np.arange(m*i, m*(i+1)) for i in xrange(0, k-c)]

    return splits
    
def set_train_sets(length):
    splits_train = get_splits(length, cpu_amount)
    i = 0
    for main_pipe_out, split_train in zip(main_pipes_out, splits_train):
        main_pipe_out.send(("train_set", (X_train[split_train], T_train[split_train])))
        i += 1

def worker_thread(pipe_in, pipe_out):
    thread_nr, nn = pipe_in.recv()

    while True:
        command, args = pipe_in.recv()

        if command == "exit":
            break
        elif command == "train_set":
            X_train, T_train = args
        elif command == "forward":
            pipe_out.send((nn.calc_feed_forward(X_train, nn.bws), ))
        elif command == "backprop":
            pipe_out.send((nn.calc_backprop(X_train, T_train, nn.bws), ))

data_size = 28

data_file = np.load("mnist_{}x{}.npz.gz".format(data_size, data_size))

X_train = data_file["X_train"]
T_train = data_file["T_train"]
X_valid = data_file["X_valid"]
T_valid = data_file["T_valid"]
X_test = data_file["X_test"]
T_test = data_file["T_test"]

print("data_size: {}".format(data_size))

nl = [X_train.shape[0], 1000, 500, T_train.shape[0]]

nn = NeuralNetwork()
nn.init_bws(X_train, T_train, nl[1:-1])

print("Single threaded")

print("Test: Calc Forward")
time_forward_start = time.time()
Y_1 = nn.calc_feed_forward(X_train, nn.bws)
time_forward_end = time.time()
print("Test: Backprop")
time_backprop_start = time.time()
bwsd_1 = nn.calc_backprop(X_train, T_train, nn.bws)
time_backprop_end = time.time()

taken_time_forward = time_forward_end-time_forward_start
taken_time_backprop = time_backprop_end-time_backprop_start

print("taken_time_forward:  {:02.5f} s".format(taken_time_forward))
print("taken_time_backprop: {:02.5f} s".format(taken_time_backprop))

print("\nMulti processing")

procs = [Process(target=worker_thread, args=(pipe_in, pipe_out)) for pipe_in, pipe_out in zip(thread_pipes_in, thread_pipes_out)]

for proc in procs:
    proc.start()

for i, main_pipe_out in enumerate(main_pipes_out):
    main_pipe_out.send((i, nn))

set_train_sets(X_train.shape[0])
print("Test: Calc Forward")
time_forward_start = time.time()
for main_pipe_out in main_pipes_out:
    main_pipe_out.send(("forward", ()))
Y2 = main_pipes_in[0].recv()[0]
# print("Y2.shape: {}".format(Y2.shape))
for main_pipe_in in main_pipes_in[1:]:
    temp = main_pipe_in.recv()[0]
    # print("temp.shape: {}".format(temp.shape))
    Y2 = np.vstack((Y2, temp))
# nn.calc_feed_forward(X_train, nn.bws)
time_forward_end = time.time()
print("Test: Backprop")
time_backprop_start = time.time()
for main_pipe_out in main_pipes_out:
    main_pipe_out.send(("backprop", ()))
bwsd_2 = main_pipes_in[0].recv()
for main_pipe_in in main_pipes_in[1:]:
    temp = main_pipe_in.recv()
    for bwsdi, t in zip(bwsd_2, temp):
        bwsdi += t
# nn.calc_backprop(X_train, T_train, nn.bws)
time_backprop_end = time.time()

taken_time_forward = time_forward_end-time_forward_start
taken_time_backprop = time_backprop_end-time_backprop_start

for main_pipe_out in main_pipes_out:
    main_pipe_out.send(("exit", ()))

for proc in procs:
    proc.join()

print("taken_time_forward:  {:02.5f} s".format(taken_time_forward))
print("taken_time_backprop: {:02.5f} s".format(taken_time_backprop))
