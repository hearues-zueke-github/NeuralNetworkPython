#! /usr/bin/python2.7

# For help: http://stackoverflow.com/questions/7194884/assigning-return-value-of-function-to-a-variable-with-multiprocessing-and-a-pr

import multiprocessing as mp
import numpy as np
import os
import pickle as cPickle #import cPickle
import gzip
import time
from NeuralNetworkMultiprocess import NeuralNetwork

if __name__ == '__main__':
    # Get amount of CPUs
    # print("CPUs count = "+str(os.cpu_count()))

    network_file_name = "learned_neural_network.txt.gz"
    inputs_targets_valid_file_name = "input_targets_valid.txt.gz"
    mnist_package_name = "mnist.pkl.gz"

    if os.path.isfile(network_file_name):
        print("Loading a created Network!")
        with gzip.open(network_file_name, 'rb') as fi:
            network = cPickle.load(fi)
        # with
        nn = network[0]
        nl = network[1]
        bs = network[2]
        ws = network[3]
        network = network[1:]
    # if

    print("nl = "+str(network[0]))

    if not os.path.isfile(inputs_targets_valid_file_name):
        with gzip.open(mnist_package_name, 'rb') as f:
            print("Loading mnist package in train_set, valid_set and test_set")
            train_set, valid_set, test_set = cPickle.load(f)
        # with

        inputs_targets_valid = [[], []]
        print("Creating and writing a file of a list of inputs and targets!")
        for inputs in valid_set[0]:
            temp = [-1 if x < 0.5 else 1 for x in inputs]
            inputs_targets_valid[0].append(np.array([inputs]).transpose())
        # for

        for targets in valid_set[1]:
            temp = [0.1 for x in xrange(0, 10)]
            temp[targets] = 0.9
            # inputs_targets_valid[1].append(np.array([temp]).transpose())
            inputs_targets_valid[1].append(targets)
        # for

        with gzip.GzipFile(inputs_targets_valid_file_name, 'wb') as fo:
            cPickle.dump(inputs_targets_valid, fo)
        # with
    else:
        print("Loading a List of inputs and targets")
        with gzip.open(inputs_targets_valid_file_name, 'rb') as fi:
            inputs_targets_valid = cPickle.load(fi)
        # with
    # if

    # Test multiprocessing
    result_queue = mp.Queue()
    
    # is_finished = []
    # Test with queue()!!!
    is_finished = mp.Manager().Queue()
    max_jobs = 100
    cpus_used = 7
    amount = 100
    args_list = [(result_queue, is_finished, i, network, inputs_targets_valid[0][i*amount:(i+1)*amount],
                  inputs_targets_valid[1][i*amount:(i+1)*amount], i*amount) for i in xrange(0, (cpus_used if cpus_used <= max_jobs else max_jobs))]

    jobs = [mp.Process(target = nn.get_approximation_guess, args=args) for args in args_list]
    # list_of_finished = [0 for i in xrange(0, cpus_used)]
    # added 20151028 jobs_not_finished
    jobs_not_finished = [1 for x in args_list]
    # for i in xrange(0, cpus_used):
    #     is_finished.put(i)
    # # for
    print("Start calculations!")
    for job in jobs: job.start()
    print("Wait for joining!")
    # error_total = 0
    result_digits = []
    index = cpus_used
    while True:
        time.sleep(1)
        proc_num = is_finished.get()
        if not proc_num is None:
            # if list_of_finished[proc_num] == 0:
                # list_of_finished[proc_num] = 1
            print("merge evaluated numbers from proc #"+str(proc_num)+" with network")
            result = result_queue.get()
            result_digits += result
            jobs_not_finished[proc_num] = 0

            if (index + 1) * amount <= max_jobs * amount:#len(inputs_targets_valid[0]):
                args = (result_queue, is_finished, proc_num, network, inputs_targets_valid[0][index*amount:(index+1)*amount],
                  inputs_targets_valid[1][index*amount:(index+1)*amount], index*amount)
                jobs[proc_num] = mp.Process(target = nn.get_approximation_guess, args=args)
                jobs_not_finished[proc_num] = 1
                jobs[proc_num].start()
                index += 1
                print("new Index: "+str(index)+"   evaluated images = "+str(index*amount))
            else:
                if jobs_not_finished.count(1) == 0:
                    break
                # if
            # if

            print("After improving:")#\n"+str(network[1]))

        else:
            print("main proc is waiting...")
        # if
    # while

    # for result in result_digits:
    #     print(str(result))
    # # for

    pos = []
    neg = []
    for result in result_digits:
        if result[1] == result[2][0][0]:
            pos.append([result[0], result[1]])
        else:
            neg.append([result[0], result[1]])

    # print("all positives:\n"+str(pos))
    # print("all negatives:\n"+str(neg))

    output_dts_name = "date_times.txt"

    if os.path.isfile(output_dts_name) is False:
        with open(output_dts_name, "w") as fout:
            fout.write("Date          "+
                       "Time        "+
                       "Pos Samples   "+
                       "Neg Samples\n")

    with open(output_dts_name, "a") as fout:
        fout.write(str(time.strftime("%Y.%m.%d    %H:%M:%S    "))+
                   str("%11d" % len(pos))+
                   str("%14d" % len(neg))+"\n")

    print("amount ofpositives:\n"+str(len(pos)))
    print("amount ofnegatives:\n"+str(len(neg)))

    print("Exit with the program!")
# if