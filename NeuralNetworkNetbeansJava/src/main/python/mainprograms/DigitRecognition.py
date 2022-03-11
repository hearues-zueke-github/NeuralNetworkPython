#! /usr/bin/python2.7

# For help: http://stackoverflow.com/questions/7194884/assigning-return-value-of-function-to-a-variable-with-multiprocessing-and-a-pr

from __init__ import *

network_file_name = "learned_neural_network.pkl.gz"
inputs_targets_train_file_name = "input_targets_train.pkl.gz"
mnist_package_name = "mnist.pkl.gz"

MAX_CPUS = mp.cpu_count() # 12

# possible smaller_factor: 1 for 28x28, 2 for 14x14, 4 for 7x7

def save_inputs_targets(inputs_targets_path, size, smaller_factor, select_set):
    # Load the training pictures from MNIST and give it to the neural network
    with gzip.open(mnist_package_name, 'rb') as f:
        print("Loading mnist package in train_set, valid_set and test_set")
        train_set, valid_set, test_set = pkl.load(f)
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
    elif select_set == "test":
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

    for targets in selected_set[1][:size]:
        temp = [0.1 for x in xrange(0, 10)]
        temp[targets] = 0.9
        inputs_targets[1].append(np.array([temp]).transpose())
        inputs_targets[2].append(targets)
    # for

    print("some values numbers: "+str(selected_set[1][:20]))

    # Save the inputs_targets
    with gzip.GzipFile(inputs_targets_path, 'wb') as fo:
        pkl.dump(inputs_targets, fo)
    # with

    return inputs_targets
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
        pkl.dump(network, fo)
    # with

    return network
# def save_new_network

# def load_network_inputs_targets(network_path, inputs_targets_path):
#     # Load the neural_network
#     print("Loading a created Network!")

#     with gzip.open(network_path, 'rb') as fi:
#         network = pkl.load(fi)
#     # with

#     # Load the inputs_targets_train, if needed later in program
#     print("Loading a List of inputs and targets")

#     with gzip.open(inputs_targets_path, 'rb') as fi:
#         inputs_targets_train = pkl.load(fi)
#     # with

#     return (network, inputs_targets_train)
# # def load_network_inputs_targets

def picture_list_to_matrix(img_1d_array, width, height, scale):
    w, h = width, height
    data = np.zeros((scale*h, scale*w, 3), dtype=np.uint8)
    img_2d_array = [list(img_1d_array[w*i:w*(i+1)]) for i in xrange(0, h)]

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

def get_picture_from_vector(img_vector, value, index, width, height, scale):
    img_1d_array = img_vector.transpose().tolist()[0]
    data = picture_list_to_matrix(img_1d_array, width, height, scale)
    img = Image.fromarray(data, "RGB")

    return img
# def get_picture_from_vector

def show_picture(img_vector, value, index, width, height, scale):
    img = get_picture_from_vector(img_vector, value, index, width, height, scale)
    img.show(title="Number "+str(value))

    print("The Digit of picture #"+str(index)+" is: "+str(value))
# def save_picture

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

    # Then load the initialized or learned network and the inputs and targets, if needed
    network = utils.load_pkl_file(network_path_input)
    inputs_targets = utils.load_pkl_file(inputs_targets_path)

    inputs = inputs_targets[0]
    targets = inputs_targets[1]
    values = inputs_targets[2]

    print("nl = "+str(network.neuronal_list))

    width = int(round(math.sqrt(len(inputs_targets[0][0].transpose().tolist()[0]))))
    height = width
    scale = 30
    pictures_amount = 5

    print("width = "+str(width))

    # Test with queue()!!!
    # Test multiprocessing
    result_queue = Queue(100)
    is_finished = Manager().Queue()
    output_queue = Queue()

    mutex = Lock()
    mutex_print = Lock()
    
    # inputs_targets[0] = inputs_targets[0][1000*1:]
    # inputs_targets[1] = inputs_targets[1][1000*1:]
    cpus_used = mp.cpu_count()
    amount = 10
    max_jobs = 100#len(inputs_targets[0])

    index = cpus_used if cpus_used * amount <= max_jobs else max_jobs / amount
    args_list = [(result_queue,
                  is_finished,
                  output_queue,
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
    # mutex_print.acquire()
    print("Start calculations!")
    # mutex_print.release()

    for job in jobs: job.start()

    # mutex_print.acquire()
    print("Wait for joining!")
    # mutex_print.release()
    # bsdt = []
    # wsdt = []
    error_total = 0

    error_list = []

    # network_orig = deepcopy(network)
    nn = [network.neuronal_list, network.biases, network.weights]
    nn_orig = deepcopy(nn)
    # bsdt, wsdt = network.get_zero_biases_weights()
    # print("biases before:\n"+str(nn[1]))

    while True:
        time.sleep(0.1)

        if not is_finished.empty():
            print("getting the result!")
            proc_num = is_finished.get()
        else:
            proc_num = None
        # if

        while not output_queue.empty():
            print(output_queue.get()),
        # while

        if not proc_num is None:
            # mutex_print.acquire()
            print("merge deltas from proc #"+str(proc_num)+" with network")
            # mutex_print.release()

            mutex.acquire()
            bsd, wsd, error_list_temp, error = result_queue.get()
            
            while not output_queue.empty():
                print(output_queue.get()),
            # while

            print("")
            mutex.release()
            
            nn = [network.neuronal_list, network.biases, network.weights]
            nn = network.get_improved_network(nn, [bsd], [wsd], copy=True)
            network.neuronal_list = nn[0]
            network.biases = nn[1]
            network.weights = nn[2]
            error_list += error_list_temp
            error_total += error

            jobs_not_finished[proc_num] = 0

            if (index + 1) * amount <= max_jobs:
                args = (result_queue,
                        is_finished,
                        output_queue,
                        mutex,
                        mutex_print,
                        proc_num,
                        network,
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
            pass
            # print("main proc is waiting...")
        # if
    # while

    print("Biases before improving:\n"+str(nn_orig[1]))
    # nn.get_improved_network(network, bsdt, wsdt)
    print("Biases after improving:\n"+str(nn[1]))

    print("error total: "+str(error_total))
    network.show_error_plot(error_list)
    print("error_list =\n"+str(error_list))

    # # print("testing functions result:\n"+str(results))
    
    print("Save the new network")

    with gzip.GzipFile(network_path_output, 'wb') as fo:
        pkl.dump(network, fo)
    # with

    print("Exit with the program!")

    return network
# def main_learning

def main_digits_testing(network_path_input, inputs_targets_path, output_statistics_path):
    network = load_pkl_file(network_path_input)
    inputs_targets = load_pkl_file(inputs_targets_path)

    size = 100
    inputs_targets = [inputs_targets[0][:size], inputs_targets[1][:size], inputs_targets[2][:size]]

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

        proc_num = is_finished.get()

        if not proc_num is None:
            print("merge evaluated numbers from proc #"+str(proc_num)+" with network")
            result = result_queue.get()

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
        else:
            print("main proc is waiting...")
        # if
    # while

    pos = []
    neg = []

    print("result[0]:"+str(result_digits[0]))

    for result in result_digits:
        if result[1] == result[2][0][0]:
            pos.append([result[0], result[1]])
        else:
            neg.append([result[0], result[1]])

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

    print("amount of positives:\n"+str(len(pos)))
    print("amount of negatives:\n"+str(len(neg)))

    print("Exit with the program!")
# def main_digits_testing

def multiprocess_network_learning(network, inputs, targets, copy=False):
    if copy:
        network = deepcopy(network)
    # if

    result_queue = Queue(1000)
    is_finished = Manager().Queue()
    # output_queue = Queue()

    mutex = Lock()
    mutex_print = Lock()

    network.set_learning_rate(0.00005)
    etha = network.get_learning_rate()
    
    cpus_used = mp.cpu_count()
    cpus_used = cpus_used if cpus_used <= MAX_CPUS else MAX_CPUS

    amount = 1000
    max_jobs = len(inputs)
    print("len(inputs) = "+str(len(inputs)))

    index = cpus_used if cpus_used * amount <= max_jobs else max_jobs / amount
    args_list = [(result_queue,
                  is_finished,
                  mutex,
                  deepcopy(network),
                  etha,
                  i,
                  inputs[i*amount:(i+1)*amount],
                  targets[i*amount:(i+1)*amount]) for i in
                  xrange(0, index)]

    print("args_list len is "+str(len(args_list)))

    mutex.acquire()
    print("before beginning, mutex acquired")
    mutex.release()
    print("after beginning, mutex released")

    jobs = [Process(target=network.get_delta_biases_weights_multiprocess, args=args) for args in args_list]
    jobs_not_finished = [1 for _ in args_list]

    print("Start calculations!")

    for job in jobs: job.start()

    print("Wait for joining!")

    error_total = 0.
    error_list = []

    nn = [network.get_neuron_list(), network.get_biases(), network.get_weights()]
    nn_orig = deepcopy(nn)

    while True:
        time.sleep(0.05)
        proc_num = None

        mutex.acquire()
        if not is_finished.empty():
            proc_num = is_finished.get()
        mutex.release()

        if not proc_num is None:
            sys.stdout.write(".")

            mutex.acquire()
            bsd, wsd, error_list_temp, error = result_queue.get()
            mutex.release()
            
            nn = [network.get_neuron_list(), network.get_biases(), network.get_weights()]
            nn = network.get_improved_network(nn, [bsd], [wsd], copy=True)
            network.neuronal_list = nn[0]
            network.biases = nn[1]
            network.weights = nn[2]
            error_list += error_list_temp
            error_total += error

            jobs_not_finished[proc_num] = 0

            if (index + 1) * amount <= max_jobs:
                args = (result_queue,
                        is_finished,
                        mutex,
                        network,
                        etha,
                        proc_num,
                        inputs[index*amount:(index+1)*amount],
                        targets[index*amount:(index+1)*amount])
                jobs[proc_num] = Process(target=network.get_delta_biases_weights_multiprocess, args=args)
                jobs_not_finished[proc_num] = 1
                jobs[proc_num].start()
                index += 1

                mutex_print.acquire()
                # print("new Index: "+str(index)+"   learned images = "+str(index*amount))
                mutex_print.release()
            else:
                if jobs_not_finished.count(1) == 0:
                    break
    print("")

    print("len(inputs) = "+str(len(inputs)))
    error_total = np.sqrt(error_total/len(inputs))

    return network, error_list, error_total

def multiprocess_network_evaluation(network, inputs, values):
    result_queue = Queue()
    is_finished = Manager().Queue()
    mutex = Lock()

    cpus_used = mp.cpu_count()
    cpus_used = cpus_used if cpus_used <= MAX_CPUS else MAX_CPUS

    amount = 100
    max_jobs = len(inputs)

    index = cpus_used if cpus_used * amount <= max_jobs else max_jobs / amount
    args_list = [(result_queue,
                  is_finished,
                  mutex,
                  i,
                  network,
                  inputs[i*amount:(i+1)*amount],
                  values[i*amount:(i+1)*amount], i*amount) for i in
                  xrange(0, index)]

    jobs = [mp.Process(target=network.get_approximation_guess, args=args) for args in args_list]
    jobs_not_finished = [1 for x in args_list]

    print("Start calculations!")
    for job in jobs: job.start()
    print("Wait for joining!")
    result_digits = []
    index = cpus_used
    while True:
        time.sleep(0.05)

        proc_num = None

        mutex.acquire()
        if not is_finished.empty():
            proc_num = is_finished.get()
        mutex.release()

        if not proc_num is None:
            sys.stdout.write(".")
            mutex.acquire()
            result = result_queue.get()
            mutex.release()

            result_digits += result
            jobs_not_finished[proc_num] = 0

            if (index + 1) * amount <= max_jobs:#len(inputs_targets_valid[0]):
                args = (result_queue, is_finished, mutex, proc_num, network, inputs[index*amount:(index+1)*amount],
                  values[index*amount:(index+1)*amount], index*amount)
                jobs[proc_num] = mp.Process(target = network.get_approximation_guess, args=args)
                jobs_not_finished[proc_num] = 1
                jobs[proc_num].start()
                index += 1
            else:
                if jobs_not_finished.count(1) == 0:
                    break
    print("")

    return result_digits
# def multiprocess_network_evaluation

def output_statistics(result_digits, error_total, learning_rate, output_path):
    pos = []
    neg = []

    print("result[0]:"+str(result_digits[0]))

    for result in result_digits:
        if result[1] == result[2][0][0]:
            pos.append([result[0], result[1]])
        else:
            neg.append([result[0], result[1]])
    # for

    if os.path.isfile(output_path) is False:
        with open(output_path, "w") as fout:
            fout.write("Date          "+
                       "Time        "+
                       "Pos Samples   "+
                       "Neg Samples            "+
                       "Error-Total     "+
                       "Learning-Rate\n")
    # if

    with open(output_path, "a") as fout:
        fout.write(str(time.strftime("%Y.%m.%d    %H:%M:%S    "))+
                   str("%11d" % len(pos))+
                   str("%14d" % len(neg))+
                   str("%22.5f" % error_total)+"%"+
                   str("%18.6f" % learning_rate)+"\n")
    # with

    print("amount of positives:\n"+str(len(pos)))
    print("amount of negatives:\n"+str(len(neg)))

    print("Finished testing network!")
# def output_statistics

def output_statistics_error(error_total, learning_rate, output_path):
    if os.path.isfile(output_path) is False:
        with open(output_path, "w") as fout:
            fout.write("Date          "+
                       "Time            "+
                       "Error-Total     "+
                       "Learning-Rate"+"\n")
    # if

    with open(output_path, "a") as fout:
        fout.write(str(time.strftime("%Y.%m.%d    %H:%M:%S    "))+
                   str("%14.5f" % error_total +"%"+
                   str("%18.6f" % learning_rate)+"\n"))
    # with

    print("Finished testing network!")
# def output_statistics_error

def multiprocess_network_learning_better(network, inputs, targets, learning_rate, copy=False):
    if copy:
        network = deepcopy(network)

    result_queue = Queue()
    network.set_learning_rate(learning_rate)
    
    cpus_used = mp.cpu_count()
    cpus_used = cpus_used if cpus_used <= MAX_CPUS else MAX_CPUS

    max_jobs = len(inputs)
    print("len(inputs) = "+str(max_jobs))

    jobs_amount_all = max_jobs / cpus_used
    jobs_amount_rest = max_jobs % cpus_used

    args = (result_queue, is_finished, mutex, deepcopy(network), learning_rate)

    args_list_1 = [args+
                  (i,
                   inputs[i*jobs_amount_all:(i+1)*jobs_amount_all],
                   targets[i*jobs_amount_all:(i+1)*jobs_amount_all]) for i in xrange(0, cpus_used - jobs_amount_rest)]

    index = (cpus_used - jobs_amount_rest) * jobs_amount_all

    args_list_2 = [args+
                  (i+(cpus_used-jobs_amount_rest),
                   inputs[index+i*(jobs_amount_all+1):index+(i+1)*(jobs_amount_all+1)],
                   targets[index+i*(jobs_amount_all+1):index+(i+1)*(jobs_amount_all+1)]) for i in xrange(0, jobs_amount_rest)]

    args_list = args_list_1 + args_list_2

    jobs = [Process(target=network.get_delta_biases_weights_multiprocess, args=args) for args in args_list]
    print("Start calculations!")
    for job in jobs: job.start()
    print("Wait for joining!")

    error_total = 0
    error_list = []

    nn = [network.get_neuron_list(), network.get_biases(), network.get_weights()]
    # TODO 2016.07.18 need also some changes too!
    for i in xrange(cpus_used):
        bsd, wsd, error_list_temp, error = result_queue.get()
        sys.stdout.write("proc #"+str(proc_num)+", ")
        nn = network.get_improved_network(nn, bsd, wsd, copy=True)
        error_list += error_list_temp
        error_total += error
    print("")

    print("len(inputs) = "+str(len(inputs)))
    error_total = np.sqrt(error_total / len(inputs))

    return network, error_list, error_total

def multiprocess_network_evaluation_binadder(network, inputs, targets, copy=False):
    if copy:
        network = deepcopy(network)

    pos = 0
    neg = 0

    result_queue = Queue(1000)
    is_finished = Manager().Queue()

    mutex = Lock()
    
    cpus_used = mp.cpu_count()
    cpus_used = cpus_used if cpus_used <= MAX_CPUS else MAX_CPUS

    max_jobs = len(inputs)
    print("len(inputs) = "+str(len(inputs)))

    jobs_amount_all = max_jobs / cpus_used
    jobs_amount_rest = max_jobs % cpus_used

    args = (result_queue, is_finished, mutex, deepcopy(network))

    args_list_1 = [args+
                  (i,
                   inputs[i*jobs_amount_all:(i+1)*jobs_amount_all],
                   targets[i*jobs_amount_all:(i+1)*jobs_amount_all]) for i in xrange(0, cpus_used - jobs_amount_rest)]

    index = (cpus_used - jobs_amount_rest) * jobs_amount_all

    args_list_2 = [args+
                  (i+(cpus_used-jobs_amount_rest),
                   inputs[index+i*(jobs_amount_all+1):index+(i+1)*(jobs_amount_all+1)],
                   targets[index+i*(jobs_amount_all+1):index+(i+1)*(jobs_amount_all+1)]) for i in xrange(0, jobs_amount_rest)]

    args_list = args_list_1 + args_list_2

    jobs = [Process(target=network.get_binary_adder_evaluation_multiprocess, args=args) for args in args_list]
    jobs_not_finished = [1 for x in args_list]

    print("Start calculations!")

    for job in jobs: job.start()

    print("Wait for joining!")

    error_total = 0

    error_list = []

    nn = [network.get_neuron_list(), network.get_biases(), network.get_weights()]
    nn_orig = deepcopy(nn)

    while True:
        time.sleep(0.05)
        proc_num = None

        mutex.acquire()
        if not is_finished.empty():
            proc_num = is_finished.get()
        # if
        mutex.release()

        if not proc_num is None:
            sys.stdout.write("proc #"+str(proc_num)+", ")

            mutex.acquire()
            pos_temp, neg_temp = result_queue.get()
            mutex.release()
            
            pos += pos_temp
            neg += neg_temp

            jobs_not_finished[proc_num] = 0

            if jobs_not_finished.count(1) == 0:
                break
            # if
        # if
    # while
    print("")

    return pos, neg

def output_statistics_binadder(pos, neg, error_total, learning_rate, output_path):
    if os.path.isfile(output_path) is False:
        with open(output_path, "w") as fout:
            fout.write("Date          "+
                       "Time        "+
                       "Pos Samples   "+
                       "Neg Samples            "+
                       "Error-Total     "+
                       "Learning-Rate\n")

    with open(output_path, "a") as fout:
        fout.write(str(time.strftime("%Y.%m.%d    %H:%M:%S    "))+
                   str("%11d" % pos)+
                   str("%14d" % neg)+
                   str("%22.5f" % error_total)+"%"+
                   str("%18.6f" % learning_rate)+"\n")

    print("amount of positives:\n"+str(pos))
    print("amount of negatives:\n"+str(neg))

    print("Finished testing network!")

def main_digits_learning_testing_iterations(network_path_input, inputs_targets_1_path, inputs_targets_2_path, network_path_output, output_statistics_path, iterations=1, is_save_error=False, is_binary=False):
    # Get amount of CPUs
    print("CPUs count = "+str(mp.cpu_count()))

    # Then load the initialized or learned network and the inputs and targets, if needed
    network = utils.load_pkl_file(network_path_input)
    inputs_targets_1 = utils.load_pkl_file(inputs_targets_1_path)
    inputs_targets_2 = utils.load_pkl_file(inputs_targets_2_path)

    size_1 = len(inputs_targets_1[0])
    size_2 = len(inputs_targets_2[0])
    
    inputs_targets_1 = [inputs_targets_1[0][:size_1], inputs_targets_1[1][:size_1], inputs_targets_1[2][:size_1]]
    inputs_targets_2 = [inputs_targets_2[0][:size_2], inputs_targets_2[1][:size_2], inputs_targets_2[2][:size_2]]

    inputs_1 = inputs_targets_1[0]
    targets_1 = inputs_targets_1[1]
    values_1 = inputs_targets_1[2]

    inputs_2 = inputs_targets_2[0]
    targets_2 = inputs_targets_2[1]
    values_2 = inputs_targets_2[2]

    print("len(inputs_1)  = "+str(len(inputs_1)))
    print("len(targets_1) = "+str(len(targets_1)))
    print("len(values_1)  = "+str(len(values_1)))

    print("len(inputs_2)  = "+str(len(inputs_2)))
    print("len(targets_2) = "+str(len(targets_2)))
    print("len(values_2)  = "+str(len(values_2)))

    print("nl = "+str(network.neuron_list))

    for i in xrange(0, iterations):
        print("Learning Iteration Nr. "+str(i))
        network, error_list, error_total = multiprocess_network_learning(network, inputs_1, targets_1)

        print("error total: "+str(error_total))

        print("Finished learning!")
        
        if not is_save_error:
            result_digits = multiprocess_network_evaluation(network, inputs_2, values_2)
            output_statistics(result_digits, error_total, network.get_learning_rate(), output_statistics_path)
        else:
            output_statistics_error(error_total, network.get_learning_rate(), output_statistics_path)
    # for

    print("Save the new network")

    utils.save_pkl_file(network, network_path_output)

    return network

def main_binary_adder_learning_testing_iterations(network_path, inputs_targets_path, learning_rate, output_statistics_path, iterations=1):
    network = utils.load_pkl_file(network_path)
    inputs, targets = utils.load_pkl_file(inputs_targets_path)

    print("len(inputs) = "+str(len(inputs)))
    print("len(targets) = "+str(len(targets)))
    print("nl = "+str(network.get_neuron_list()))

    for i in xrange(0, iterations):
        print("Learning Iteration Nr. "+str(i))
        network, error_list, error_total = multiprocess_network_learning_better(network, inputs, targets, learning_rate)

        print("error total: "+str(error_total))
        print("Finished Learning!")

        pos, neg = multiprocess_network_evaluation_binadder(network, inputs, targets)
        print("pos = "+str(pos))
        print("neg = "+str(neg))
        output_statistics_binadder(pos, neg, error_total, learning_rate, output_statistics_path)

    print("Save the new network\n")
    utils.save_pkl_file(network, network_path)
    return network

def main_binary_adder_learning_testing_iterations_better(network_path, inputs_targets_path, learning_rate, output_statistics_path, iterations=1):
    network = utils.load_pkl_file(network_path)
    inp_train, targ_train, inp_test, targ_test = utils.load_pkl_file(inputs_targets_path)

    print("len(inputs) = "+str(len(inp_train)))
    print("len(targets) = "+str(len(targ_train)))
    print("len(inputs) = "+str(len(inp_test)))
    print("len(targets) = "+str(len(targ_test)))
    print("neurons per layer = "+str(network.get_neuron_list()))

    for i in xrange(0, iterations):
        print("Learning Iteration Nr. "+str(i))
        network, error_list, error_total = multiprocess_network_learning_better(network, inp_train, targ_train, learning_rate)

        print("error total: "+str(error_total))
        print("Finished Learning!")

        pos, neg = multiprocess_network_evaluation_binadder(network, inp_test, targ_test)
        print("pos = "+str(pos))
        print("neg = "+str(neg))
        output_statistics_binadder(pos, neg, error_total, learning_rate, output_statistics_path)

    print("Save the new network\n")
    utils.save_pkl_file(network, network_path)
    return network

if __name__ == '__main__' or __name__ == "DigitRecognition":
    pass
