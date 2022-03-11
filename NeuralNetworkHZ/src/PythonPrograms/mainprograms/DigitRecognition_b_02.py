#! /usr/bin/python2.7

# For help: http://stackoverflow.com/questions/7194884/assigning-return-value-of-function-to-a-variable-with-multiprocessing-and-a-pr

# import multiprocessing as mp
import os
import gzip
import time
import sys
import math

import multiprocessing as mp
import numpy as np
import pickle as cPickle

from NeuralNetworkDecimalMultiprocess import NeuralNetwork
from multiprocessing import Queue, Manager, Process, Lock
from copy import deepcopy
from PIL import Image

# print("__name__ = "+str(__name__))

network_file_name = "learned_neural_network.pkl.gz"
inputs_targets_train_file_name = "input_targets_train.pkl.gz"
mnist_package_name = "mnist.pkl.gz"

# possible smaller_factor: 1 for 28x28, 2 for 14x14, 4 for 7x7

def save_inputs_targets(inputs_targets_path, size, smaller_factor, select_set):
    # Load the training pictures from MNIST and give it to the neural network
    with gzip.open(mnist_package_name, 'rb') as f:
        print("Loading mnist package in train_set, valid_set and test_set")
        train_set, valid_set, test_set = cPickle.load(f)
    # with

    # train_set_len_orig = len(train_set[0])
    train_set_len = size# train_set_len_orig

    train_set = [train_set[0][:train_set_len], train_set[1][:train_set_len]]
    valid_set = [valid_set[0][:train_set_len], valid_set[1][:train_set_len]]
    test_set = [test_set[0][:train_set_len], test_set[1][:train_set_len]]

    inputs_targets = [[], [], []]
    print("Creating and writing a file of a list of inputs and targets!")
    if select_set == "train":
        selected_set = train_set
    elif select_set == "valid":
        selected_set = valid_set
    elif select_set == "DrawNumbers":
        selected_set = test_set
    else:
        return None
    # if

    # original size is 28x28
    for inputs in selected_set[0][:size]:
        temp = [0 if x < 0.5 else 1 for x in inputs]
        inputs_targets[0].append(np.array([temp]).transpose())
    # for
    print("")

    # print("smaller_factor = "+str(smaller_factor)+"   type = "+str(type(smaller_factor)))
    smaller_factor = float(smaller_factor)
    dict_factors = [1., 2., 4.]
    for df1, df2 in zip(dict_factors[:-1], dict_factors[1:]):
        if smaller_factor - df1 < (df2 - df1) / 2.:
            smaller_factor = df1
            break
        # if
    # for
    if smaller_factor > dict_factors[-1]:
        smaller_factor = dict_factors[-1]
    # if

    sf = float(smaller_factor)
    smaller_factor = int(smaller_factor)

    inputs_smaller = []
    for inputs in inputs_targets[0][:size]:
        temp = inputs.transpose().tolist()[0]
        temp_2d = [temp[i*28:(i+1)*28] for i in xrange(0, 28)]

        temp_2d_small = [[0 for _ in xrange(0, 28 / smaller_factor)] for _ in xrange(0, 28 / smaller_factor)]

        for y in xrange(0, 28 / smaller_factor):
            for x in xrange(0, 28 / smaller_factor):
                for y1 in xrange(0, smaller_factor):
                    for x1 in xrange(0, smaller_factor):
                        temp_2d_small[y][x] += temp_2d[smaller_factor*y+y1][smaller_factor*x+x1]
                    # for
                # for
                temp_2d_small[y][x] /= sf**2
            # for
        # for

        temp_small = []
        for t in temp_2d_small:
            temp_small += t
        # for

        if len(inputs_smaller) == 0:
            for tmp in temp_2d:
                print(str(tmp))
            # for

            for tmp in temp_2d_small:
                print(str(tmp))
            # for

            # print(str(temp_2d_small))
        # if

        inputs_smaller.append(np.array([temp_small]).transpose())
    # for

    inputs_targets[0] = inputs_smaller

    # TODO: define a function, which shrink the picture

    for targets in train_set[1][:size]:
        temp = [0.1 for x in xrange(0, 10)]
        temp[targets] = 0.9
        inputs_targets[1].append(np.array([temp]).transpose())
        inputs_targets[2].append(targets)
    # for

    # Save the inputs_targets
    with gzip.GzipFile(inputs_targets_path, 'wb') as fo:
        cPickle.dump(inputs_targets, fo)
    # with

    return input_targets_train
# def save_inputs_targets

def save_new_network(network_path, inputs_targets, smaller_factor):
    # # print(str(train_set[0][0]))
    # width = int(round(math.sqrt(len(inputs_targets_train[0][0].transpose().tolist()[0]))))
    # height = width
    # # print("before:\nwidth = "+str(width)+"     height = "+str(height))
    # scale = 30
    # pictures_amount = 5

    # inputs_list = [inputs_targets_train[0][i].transpose().tolist()[0] for i in xrange(0, pictures_amount)]
    # values_list = [inputs_targets_train[2][i] for i in xrange(0, pictures_amount)]

    # for inp, val, index in zip(inputs_list, values_list, xrange(0, len(inputs_list))):
    #     show_picture(inp, val, index, width, height, scale)
    # # for

    print("Creating a new Network!")
    network = NeuralNetwork()
    # nl = [28*28, 28*28+28*14, 10]
    # bs = nn.get_random_biases_list(nl)
    # ws = nn.get_random_weight_list(nl)

    # network = [nn, nl, bs, ws]

    # Initialize the whole network
    #[28*28, 28*28+28*14, 10],
    layer_factor = 28 / smaller_factor
    network.init_complete_network([layer_factor**2, layer_factor**2+layer_factor**2 / 2, 10],
                                  inputs_targets[0],#inputs_smaller,
                                  inputs_targets[1])#targets)
    network.targets_values = inputs_targets[2]
    print("network.neuronal_list = "+str(network.neuronal_list))

    # Save the neural_network
    with gzip.GzipFile(network_path, 'wb') as fo:
        cPickle.dump(network, fo)
    # with

    return network
# def create_new_network

def load_pkl_file(file_path):
    # Load the neural_network
    print("Loading file "+file_path)

    with gzip.open(file_path, 'rb') as fi:
        obj = cPickle.load(fi)
    # with

    return obj
# def load_network_inputs_targets

# def load_network_inputs_targets(network_path, inputs_targets_path):
#     # Load the neural_network
#     print("Loading a created Network!")

#     with gzip.open(network_path, 'rb') as fi:
#         network = cPickle.load(fi)
#     # with

#     # Load the inputs_targets_train, if needed later in program
#     print("Loading a List of inputs and targets")

#     with gzip.open(inputs_targets_path, 'rb') as fi:
#         inputs_targets_train = cPickle.load(fi)
#     # with

#     return (network, inputs_targets_train)
# # def load_network_inputs_targets

def picture_list_to_matrix(img_1d_array, width, height, scale):
    w, h = width, height
    # scale = 20
    print("scale factor in picture list to matrix = "+str(scale))

    data = np.zeros((scale*h, scale*w, 3), dtype=np.uint8)
        
    # print("after:\nwidth = "+str(width)+"     height = "+str(height))

    img_2d_array = [list(img_1d_array[w*i:w*(i+1)]) for i in xrange(0, h)]
    # print("img_2d_array = "+str(img_2d_array))
    # print("img_array = "+str(img_array))

    # print("img_1d_array =\n"+str(img_1d_array))
    # print("img_2d_array =\n"+str(img_2d_array))

    # print("len y = "+str(len(img_2d_array))+"    len x = "+str(len(img_2d_array[0])))
    # print("img_2d_array =\n"+str(img_2d_array))
    for y in xrange(0, h):
        for x in xrange(0, w):
            # print("x = "+str(x)+"   y = "+str(y))
            color_value = int(float(img_2d_array[y][x]) * 255.)
            color = (color_value, color_value, color_value)
            for ys in xrange(0, scale):
                for xs in xrange(0, scale):
                    data[scale*y+ys, scale*x+xs] = color
                # for
            # for
        # for
    # for

    return data
# def pixels_list_to_2d_list

def show_picture(img_1d_array, value, index, width, height, scale):
    # w, h = 28, 28
    # w, h = 14, 14

    print("scale factor in show = "+str(scale))
    data = picture_list_to_matrix(img_1d_array, width, height, scale)
    img = Image.fromarray(data, "RGB")
    img.show(title="Number "+str(value))
    # img.save(picture_directory+"train_"+str(index)+"_num_"+str(list_pixels[2][index])+".png")

    print("The Digit of picture #"+str(index)+" is: "+str(value))
# def sace_pictures_in_directory

def save_pictures_in_directory(imgs_1d_array, values_1d_array, width, height, scale, picture_directory):
    w, h = 28, 28
    scale = 4

    data = np.zeros((scale*h, scale*w, 3), dtype=np.uint8)
    # rgb_white = (255, 255, 255)
    # rgb_black = (0, 0, 0)
    # print("type of 0 element: "+str(type(inputs_targets_train[0])))
    for index in xrange(0, len(imgs_1d_array[0])):
        data = picture_list_to_matrix(imgs_1d_array[index], width, height, scale)
        img = Image.fromarray(data, "RGB")
        
        img.save(picture_directory+"train_"+str(index)+"_num_"+str(values_1d_array[index])+".png")

        print("The Digit of picture #"+str(index)+" is: "+str(values_1d_array[index]))
    # for
# def sace_pictures_in_directory

def main_digits_learning(network_path_input, inputs_targets_path, network_path_output):
    # Get amount of CPUs
    print("CPUs count = "+str(mp.cpu_count()))

    print("TEst!!!")

    # # First, check if the learned network exists!
    # if not os.path.isfile(network_file_name):
    #     create_new_network()
    # # if

    # Then load the initialized or learned network and the inputs and targets, if needed
    network = load_pkl_file(network_path_input)
    inputs_targets = load_pkl_file(inputs_targets_path)

    inputs = inputs_targets[0]
    targets = inputs_targets[1]
    values = inputs_targets[2]

    print("nl = "+str(network.neuronal_list))

    # set the picture at the index as a png
    # index = 10
    # save_pictures_in_directory(inputs_targets,
    #                            "/home/haris/Pictures/mnist_pictures_size_28x4/")
    # sys.exit(0)

    width = int(round(math.sqrt(len(inputs_targets[0][0].transpose().tolist()[0]))))
    height = width
    scale = 30
    pictures_amount = 5

    print("width = "+str(width))
    
    inputs_list = [inputs[i].transpose().tolist()[0] for i in xrange(0, pictures_amount)]
    values_list = [values[i] for i in xrange(0, pictures_amount)]

    for inp, val, index in zip(inputs_list, values_list, xrange(0, pictures_amount)):
        show_picture(inp, val, index, width, height, scale)
    # for

    # show_picture(inputs_list[0], values_list[0], width, height, scale)
    # show_picture(inputs_list[1], values_list[1], width, height, scale)
    # show_picture(inputs_list[2], values_list[2], width, height, scale)
    # show_picture(inputs_list[3], values_list[3], width, height, scale)

    # sys.exit(0)

    # Test with queue()!!!
    # Test multiprocessing
    result_queue = Queue()
    is_finished = Manager().Queue()

    mutex = Lock()
    mutex_print = Lock()
    
    # inputs_targets[0] = inputs_targets[0][1000*1:]
    # inputs_targets[1] = inputs_targets[1][1000*1:]
    cpus_used = mp.cpu_count()
    amount = 50
    max_jobs = len(inputs_targets[0])

    index = cpus_used if cpus_used * amount <= max_jobs else max_jobs / amount
    args_list = [(result_queue,
                  is_finished,
                  mutex,
                  mutex_print,
                  i,
                  network,
                  inputs[i*amount:(i+1)*amount],
                  targets[i*amount:(i+1)*amount],
                  0.05) for i in
                  xrange(0, index)]

    print("args_list len is "+str(len(args_list)))

    mutex.acquire()
    print("before beginning, mutex acquired")
    mutex.release()
    print("after beginning, mutex released")

    jobs = [Process(target=network.get_delta_biases_weights_multiprocess, args=args) for args in args_list]
    # list_of_finished = [0 for i in xrange(0, cpus_used)]
    # added 20151028 jobs_not_finished
    jobs_not_finished = [1 for x in args_list]
    mutex_print.acquire()
    print("Start calculations!")
    mutex_print.release()

    for job in jobs: job.start()

    mutex_print.acquire()
    print("Wait for joining!")
    mutex_print.release()
    # bsdt = []
    # wsdt = []
    error_total = 0

    network_orig = deepcopy(network)
    nn = [network.neuronal_list, network.biases, network.weights]
    nn_orig = deepcopy(nn)
    # bsdt, wsdt = network.get_zero_biases_weights()
    # print("biases before:\n"+str(nn[1]))

    while True:
        # mutex_print.acquire()
        print("dead lock!!!")
        # mutex_print.release()

        time.sleep(1)

        # mutex.acquire()
        # mutex_print.acquire()
        print("getting the result!")
        # mutex_print.release()

        proc_num = is_finished.get()
        # mutex.release()

        if not proc_num is None:
            mutex_print.acquire()
            print("merge deltas from proc #"+str(proc_num)+" with network")
            mutex_print.release()

            mutex.acquire()
            bsd, wsd, error = result_queue.get()
            mutex.release()
            
            network.get_improved_network(nn, [bsd], [wsd])
            error_total += error

            jobs_not_finished[proc_num] = 0

            if (index + 1) * amount <= max_jobs:
                args = (result_queue,
                        is_finished,
                        mutex,
                        mutex_print,
                        proc_num,
                        network_orig,
                        inputs[index*amount:(index+1)*amount],
                        targets[index*amount:(index+1)*amount],
                        0.05)
                jobs[proc_num] = Process(target=network.get_delta_biases_weights_multiprocess, args=args)
                jobs_not_finished[proc_num] = 1
                jobs[proc_num].start()
                index += 1

                mutex_print.acquire()
                print("new Index: "+str(index)+"   learned images = "+str(index*amount))
                mutex_print.release()
            else:
                if jobs_not_finished.count(1) == 0:
                    break
                # if
            # if
        else:
            print("main proc is waiting...")
        # if
    # while

    print("Biases before improving:\n"+str(nn_orig[1]))
    # nn.get_improved_network(network, bsdt, wsdt)
    print("Biases after improving:\n"+str(nn[1]))

    # # print("testing functions result:\n"+str(results))
    
    print("Save the new network")

    with gzip.GzipFile(network_path_output, 'wb') as fo:
        cPickle.dump(network, fo)
    # with

    print("Exit with the program!")
# def main_learning

def main_digits_testing(network_path_input, inputs_targets_path, output_statistics_path):
     # Get amount of CPUs
    # print("CPUs count = "+str(os.cpu_count()))

    # if os.path.isfile(network_file_name):
    #     print("Loading a created Network!")
    #     with gzip.open(network_file_name, 'rb') as fi:
    #         network = cPickle.load(fi)
    #     # with
    #     nn = network[0]
    #     nl = network[1]
    #     bs = network[2]
    #     ws = network[3]
    #     network = network[1:]
    # # if

    # if not os.path.isfile(inputs_targets_valid_file_name):
    #     with gzip.open(mnist_package_name, 'rb') as f:
    #         print("Loading mnist package in train_set, valid_set and test_set")
    #         train_set, valid_set, test_set = cPickle.load(f)
    #     # with

    #     inputs_targets_valid = [[], []]
    #     print("Creating and writing a file of a list of inputs and targets!")
    #     for inputs in valid_set[0]:
    #         temp = [-1 if x < 0.5 else 1 for x in inputs]
    #         inputs_targets_valid[0].append(np.array([inputs]).transpose())
    #     # for

    #     for targets in valid_set[1]:
    #         temp = [0.1 for x in xrange(0, 10)]
    #         temp[targets] = 0.9
    #         # inputs_targets_valid[1].append(np.array([temp]).transpose())
    #         inputs_targets_valid[1].append(targets)
    #     # for

    #     with gzip.GzipFile(inputs_targets_valid_file_name, 'wb') as fo:
    #         cPickle.dump(inputs_targets_valid, fo)
    #     # with
    # else:
    #     print("Loading a List of inputs and targets")
    #     with gzip.open(inputs_targets_valid_file_name, 'rb') as fi:
    #         inputs_targets_valid = cPickle.load(fi)
    #     # with
    # # if

    network = load_pkl_file(network_path_input)
    inputs_targets = load_pkl_file(inputs_targets_path)

    inputs_targets = [inputs_targets[0][:1000], inputs_targets[1][:1000], inputs_targets[2][:1000]]

    inputs = inputs_targets[0]
    targets = inputs_targets[1]
    values = inputs_targets[2]

    print("nl = "+str(network.neuronal_list))
    
    # Test multiprocessing

    # Test with queue()!!!
    result_queue = Queue()
    is_finished = Manager().Queue()
    mutex = Lock()

    max_jobs = len(inputs_targets[0])
    cpus_used = mp.cpu_count()
    amount = 100

    index = cpus_used if cpus_used * amount <= max_jobs else max_jobs / amount
    args_list = [(result_queue,
                  is_finished,
                  mutex,
                  i,
                  network,
                  inputs[i*amount:(i+1)*amount],
                  values[i*amount:(i+1)*amount], i*amount) for i in
                  xrange(0, index)]

    jobs = [mp.Process(target = network.get_approximation_guess, args=args) for args in args_list]
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
        # with mutex:
        proc_num = is_finished.get()
        # # with

        if not proc_num is None:
            # if list_of_finished[proc_num] == 0:
                # list_of_finished[proc_num] = 1
            print("merge evaluated numbers from proc #"+str(proc_num)+" with network")
            # with mutex:
            result = result_queue.get()
            # print("result =\n"+str(result))
            # # with

            result_digits += result
            jobs_not_finished[proc_num] = 0

            if (index + 1) * amount <= max_jobs:#len(inputs_targets_valid[0]):
                args = (result_queue, is_finished, mutex, proc_num, network, inputs[index*amount:(index+1)*amount],
                  values[index*amount:(index+1)*amount], index*amount)
                jobs[proc_num] = mp.Process(target = network.get_approximation_guess, args=args)
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

    # output_dts_name = "date_times.txt"

    if os.path.isfile(output_statistics_path) is False:
        with open(output_statistics_path, "w") as fout:
            fout.write("Date          "+
                       "Time        "+
                       "Pos Samples   "+
                       "Neg Samples\n")

    with open(output_statistics_path, "a") as fout:
        fout.write(str(time.strftime("%Y.%m.%d    %H:%M:%S    "))+
                   str("%11d" % len(pos))+
                   str("%14d" % len(neg))+"\n")

    print("amount ofpositives:\n"+str(len(pos)))
    print("amount ofnegatives:\n"+str(len(neg)))

    print("Exit with the program!")
# def main_digits_testing

if __name__ == '__main__' or __name__ == "DigitRecognition":
    pass
# if
