#! /usr/bin/python2.7

# For help: http://stackoverflow.com/questions/7194884/assigning-return-value-of-function-to-a-variable-with-multiprocessing-and-a-pr

import multiprocessing as mp
import numpy as np
import os
import pickle as cPickle #import cPickle
import gzip
import time
from NeuralNetworkMultiprocess import NeuralNetwork

# # Calc the sum from to x
# def f(x, que):
#     s = 0
#     for i in range(1, x + 1):
#         s += i
#     que.put(s)

if __name__ == '__main__':
    # Get amount of CPUs
    # print("CPUs count = "+str(os.cpu_count()))

    network_file_name = "learned_neural_network.txt.gz"
    inputs_targets_file_name = "input_targets.txt.gz"

    if not os.path.isfile(network_file_name):
        nn = NeuralNetwork()
        print("Creating a new Network!")
        nl = [28*28, 28*28*2, 10]
        bs = nn.get_random_biases_list(nl)
        ws = nn.get_random_weight_list(nl)

        network = [nn, nl, bs, ws]

        with gzip.GzipFile(network_file_name, 'wb') as fo:
            cPickle.dump(network, fo)

        network = network[1:]
    else:
        with gzip.open(network_file_name, 'rb') as fi:
            network = cPickle.load(fi)
        nn = network[0]
        nl = network[1]
        bs = network[2]
        ws = network[3]
        network = network[1:]

        print("Loading a created Network!")

    print("nl = "+str(network[0]))

    with gzip.open('mnist.pkl.gz', 'rb') as f:
        print("Loading mnist package in train_set, valid_set and test_set")
        train_set, valid_set, test_set = cPickle.load(f)
    
    if not os.path.isfile(inputs_targets_file_name):
        inputs_targets = [[], []]
        print("Creating and writing a file of a list of inputs and targets!")
        for inputs in test_set[0]:
            inputs_targets[0].append(np.array([inputs]).transpose())

        for targets in test_set[1]:
            temp = [0.1 for x in xrange(0, 10)]
            temp[targets] = 0.9
            inputs_targets[1].append(np.array([temp]).transpose())

        # print(str(inputs_targets[0][0]))
        # print(str(inputs_targets[1][0]))

        with gzip.GzipFile(inputs_targets_file_name, 'wb') as fo:
            cPickle.dump(inputs_targets, fo)
    else:
        print("Loading a List of inputs and targets")
        with gzip.open(inputs_targets_file_name, 'rb') as fi:
            inputs_targets = cPickle.load(fi)

    # print(str(inputs_targets[0][0]))
    # print(str(inputs_targets[1][0]))

    # Test multiprocessing
    result_queue = mp.Queue()
    # def get_delta_biases_weights(self, queue, network, inputs, targets, etha, iterations):#, show_debug_msg = False):
    # def get_improved_network(self, network, etha, bsd, wsd):
    # locks_list = [mp.Lock()]
    
    # is_finished = []
    # Test with queue()!!!
    is_finished = mp.Manager().Queue()
    cpus_used = 3
    amount = 10
    args_list = [(result_queue, is_finished, i, network, inputs_targets[0][i*amount:(i+1)*amount],
                  inputs_targets[1][i*amount:(i+1)*amount], 0.005) for i in xrange(0, cpus_used)]
                 #(result_queue, is_finished, 1, network, inputs_targets[0][10:20], inputs_targets[1][10:20], 0.005)]#,
                 # (result_queue, network, inputs_targets[0][20:30], inputs_targets[1][20:30], 0.005)]
    # jobs = [0] * cpus_used
    jobs = [mp.Process(target = nn.get_delta_biases_weights, args=args) for args in args_list]
   
    print("Start calculations!")
    for job in jobs: job.start()
    print("Wait for joining!")
    # list_of_finished = [0 for i in xrange(0, cpus_used)]
    error_total = 0
    index = cpus_used
    while True:
        time.sleep(1)
        return_value = is_finished.get()
        print(str(type(return_value))
        if not return_value is None:
            # if list_of_finished[proc_num] == 0:
                # list_of_finished[proc_num] = 1
            print("merge deltas from proc #"+str(proc_num)+" with network")
            bsd, wsd, error = result_queue.get()
            error_total += error
            print("Before improving:")#\n"+str(network[1]))
            nn.get_improved_network(network, bsd, wsd)

            if (index + 1) * amount <= len(inputs_targets[0]):
                args = (result_queue, is_finished, proc_num, is_last, network, inputs_targets[0][index*amount:(index+1)*amount],
                  inputs_targets[1][index*amount:(index+1)*amount], 0.005)
                jobs[proc_num] = mp.Process(target = nn.get_delta_biases_weights, args=args)
                jobs[proc_num].start()
                index += 1
                print("new Index: "+str(index)+"   learned images = "+str(index*amount))
            else:
                break

            print("After improving:")#\n"+str(network[1]))
        # print("proc # to do: "+str(list_of_finished))
        else:
            print("main proc is waiting...")
            # count = 0
            # for lof in list_of_finished:
            #     if lof == 1:
            #         count += 1
            # if count == len(list_of_finished):
            #     print("All processes finished!")
            #     break

            
    # for job in jobs: job.join()
    # while True:
    #     count = 0
    #     for isf in is_finished:
    #         if isf[0] is True:
    #             count += 1

    # print("Get all returns")
    # results = [result_queue.get() for j in jobs]
    # print("Merge all deltas together")

    # bsd = []
    # wsd = []
    # errors = 0
    # for r in results:
    #     bsd += r[0]
    #     wsd += r[1]
    #     errors += r[2]

    print("Before improving:\n"+str(network[1]))
    nn.get_improved_network(network, bsd, wsd)
    print("After improving:\n"+str(network[1]))

    # # print("testing functions result:\n"+str(results))
    
    print("Save the new network")

    network = [nn, nl, bs, ws]
    with gzip.GzipFile(network_file_name, 'wb') as fo:
        cPickle.dump(network, fo)
    network = network[1:]

    print("Exit with the program!")
