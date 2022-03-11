#! /usr/bin/python2.7

from __init__ import *

PREFIX_DIGIT_RECOGNIZER = "digit_recognizer"
PREFIX_BINARY_ADDER = "binary_adder"
def init_network(network_path, nl=None, max_etha=10.**0, min_etha=10.**-4, adapt_learn_pos=1.02, adapt_learn_neg=0.7,
                 neural_name="", type_learning="sgd", hidden_function="sigmoid",
                 is_adaptive=True, is_no_random=False, network_type="regression"):
    utils.check_create_dir(network_path)

    nn = NeuralNetwork()
    nn.set_neuron_list(nl)
    nn.init_random_weights()
    nn.set_max_min_etha(max_etha, min_etha)
    nn.set_adapt_speed(adapt_learn_pos, adapt_learn_neg)
    nn.set_hidden_function(hidden_function)
    nn.set_network_type(network_type)
    nn.set_name_of_neural_network(neural_name)
    nn.set_type_of_learning(type_learning)
    nn.set_is_adaptive(is_adaptive)
    nn.set_is_no_random(is_no_random)

    utils.save_pkl_file(nn, network_path+"nn.pkl.gz")
    utils.save_pkl_file(nn, network_path+"nn_init.pkl.gz")

    return nn

def init_network_statistics(network_path, nn, inp_tr, targ_tr, inp_tst, targ_tst, first_etha=0.001):
    print("inputs_train = {}".format(inp_tr))
    print("targets_train = {}".format(targ_tr))
    print("inputs_test = {}".format(inp_tst))
    print("targets_test = {}".format(targ_tst))

    start_time = time.time()
    utils.save_pkl_file(inp_tr,   network_path+"inputs_train.pkl.gz")
    print("needed time for loading inp_tr: {}".format(time.time() - start_time))

    start_time = time.time()
    utils.save_pkl_file(targ_tr,  network_path+"targets_train.pkl.gz")
    print("needed time for loading targ_tr: {}".format(time.time() - start_time))

    start_time = time.time()
    utils.save_pkl_file(inp_tst,  network_path+"inputs_test.pkl.gz")
    print("needed time for loading inp_tst: {}".format(time.time() - start_time))

    start_time = time.time()
    utils.save_pkl_file(targ_tst, network_path+"targets_test.pkl.gz")
    print("needed time for loading targ_tst: {}".format(time.time() - start_time))

    nl, bs, ws = nn.get_neuron_list(), nn.get_biases(), nn.get_weights()

    start_time = time.time()
    results = nn.calc_forward_better_multiprocess([nl, bs, ws], inp_tr, targ_tr, inp_tst, targ_tst)
    print("needed time for nn.calc_forward_better_multiprocess: {}s".format(time.time() - start_time))
    error_train = results[0]
    error_test = results[1]
    error_train_cecf = results[2]
    error_test_cecf = results[3]
    class_train_true = results[4]
    class_test_true = results[5]

    statistics_map = {"errors_train":       [1], #[error_train],
                      "errors_test":        [1], #[error_test],
                      "errors_train_cecf":  [1], #[error_train_cecf],
                      "errors_test_cecf":   [1], #[error_test_cecf],
                      "etha_per_iteration": [[], []],
                      "iterations":         0,
                      "last_etha":          first_etha,
                      "classes_train_true": [class_train_true],
                      "classes_test_true":  [class_test_true],
                      "weights_min":        [[] for _ in nl[:-1]],
                      "weights_max":        [[] for _ in nl[:-1]],
                      "biases_min":         [[] for _ in nl[:-1]],
                      "biases_max":         [[] for _ in nl[:-1]]}
    print("start error train: {}".format(statistics_map["errors_train"][0]))
    print("start error test: {}".format(statistics_map["errors_test"][0]))
    utils.save_pkl_file(statistics_map, network_path+"statistics_map.pkl.gz")

def create_digit_recognizer_neural_network(network_name, neural_list=None, train_amount=100, test_amount=100,
                                           max_etha=10.**-3, min_etha=10.**-6, adapt_learn_pos=1.02, adapt_learn_neg=0.7,
                                           hidden_function="sigmoid", type_learning="sgd", is_adaptive=True, is_no_random=False):
    network_path = "networks/"+network_name+"/"

    nn = init_network(network_path, nl=neural_list, max_etha=max_etha, min_etha=min_etha,
                      adapt_learn_pos=adapt_learn_pos, adapt_learn_neg=adapt_learn_neg, neural_name="digit_recognizer",
                      type_learning=type_learning, hidden_function=hidden_function, is_adaptive=is_adaptive,
                      is_no_random=is_no_random, network_type="classifier")

    inp_tr, targ_tr = utils.load_pkl_file("original_sets/train_set.pkl.gz")
    inp_tst, targ_tst = utils.load_pkl_file("original_sets/test_set.pkl.gz")

    if train_amount != inp_tr.shape[0] and test_amount != inp_tr.shape[0]:
        random_indices_tr = np.random.permutation(np.arange(inp_tr.shape[0]))[:train_amount]
        random_indices_tst = np.random.permutation(np.arange(inp_tst.shape[0]))[:test_amount]

        inp_tr, targ_tr = inp_tr[random_indices_tr], targ_tr[random_indices_tr]
        inp_tst, targ_tst = inp_tst[random_indices_tst], targ_tst[random_indices_tst]

    init_network_statistics(network_path, nn, inp_tr, targ_tr, inp_tst, targ_tst, min_etha)

def create_binary_adder_neural_network(network_name, bits=4, max_etha=10.**0, min_etha=10.**-4, type_learning="sgd", hidden_function="sigmoid", first_etha=0.005, is_sgd=True, is_adaptive=True, is_no_random=False):
    """ This will create a new Network for the binary adder """
    # network_name = "neural_network_binary_adder_"+str(bits)+"_bits"+"_"+name_suffix
    network_path = "networks/"+network_name+"/"

    nn = init_network(network_path, [bits*2, bits*4, bits*4, bits+1], max_etha, min_etha, neural_name="binary_adder", type_learning=type_learning, hidden_function=hidden_function, is_sgd=is_sgd, is_adaptive=is_adaptive, is_no_random=is_no_random)

    inp_tr, targ_tr, inp_tst, targ_tst = binadd.get_binaryadder_train_test_sets(bits, True)

    init_network_statistics(network_path, nn, inp_tr, targ_tr, inp_tst, targ_tst, first_etha=first_etha)

def create_random_network(network_name, train_amount, test_amount):
    network_path = "networks/"+network_name+"/"
    nl = [3, 10, 1]
    nn = init_network(network_path, nl)

    inp_tr = np.random.random((train_amount, nl[0]))
    targ_tr = np.random.random((train_amount, nl[-1]))*0.8+0.1
    inp_tst = np.random.random((test_amount, nl[0]))
    targ_tst = np.random.random((test_amount, nl[-1]))*0.8+0.1
    bs, ws = nn.get_biases(), nn.get_weights()
    # targ_tr = nn.calculate_forward_many(inp_tr, bs, ws)
    # targ_tst = nn.calculate_forward_many(inp_tst, bs, ws)

    nn = init_network(network_path, nl)
    init_network_statistics(network_path, nn, inp_tr, targ_tr, inp_tst, targ_tst, 0.0001)

def learn_neural_network(network_name, iterations=10, with_sgd=True, with_adaptive=True, with_no_random=False):
    network_path = "networks/"+network_name+"/"

    nn = utils.load_pkl_file(network_path+"nn.pkl.gz")
    inp_tr = utils.load_pkl_file(network_path+"inputs_train.pkl.gz")
    targ_tr = utils.load_pkl_file(network_path+"targets_train.pkl.gz")
    inp_tst = utils.load_pkl_file(network_path+"inputs_test.pkl.gz")
    targ_tst = utils.load_pkl_file(network_path+"targets_test.pkl.gz")

    statistics_map = utils.load_pkl_file(network_path+"statistics_map.pkl.gz")
    errors_train = statistics_map["errors_train"]
    errors_test = statistics_map["errors_test"]
    errors_train_cecf = statistics_map["errors_train_cecf"]
    errors_test_cecf = statistics_map["errors_test_cecf"]
    etha_per_iteration = statistics_map["etha_per_iteration"]
    iterations_sum = statistics_map["iterations"]
    last_etha = statistics_map["last_etha"]
    classes_train_true = statistics_map["classes_train_true"]
    classes_test_true = statistics_map["classes_test_true"]

    network_type = nn.get_network_type()

    nl, bs, ws = nn.get_neuron_list(), nn.get_biases(), nn.get_weights()

    components, errors_train_calc, errors_test_calc, \
                errors_train_cecf_calc, errors_test_cecf_calc, \
                etha_per_iteration_calc,\
                classes_train_true_calc, classes_test_true_calc, \
                max_wsmods, max_bsmods, min_wsmods, min_bsmods = \
        nn.improve_network_multiprocess([nl, bs, ws], inp_tr, targ_tr,
                                        inp_tst, targ_tst, iterations, last_etha,
                                        copy=True, withrandom=False, start_iter=iterations_sum)
    nn.set_network(components) # components = [nl, bs, ws]
    errors_train.extend(errors_train_calc[1:])
    errors_test.extend(errors_test_calc[1:])
    errors_train_cecf.extend(errors_train_cecf_calc[1:])
    errors_test_cecf.extend(errors_test_cecf_calc[1:])
    etha_per_iteration[0].extend(etha_per_iteration_calc[0])
    etha_per_iteration[1].extend(etha_per_iteration_calc[1])
    classes_train_true.extend(classes_train_true_calc[1:])
    classes_test_true.extend(classes_test_true_calc[1:])

    extend_lists = lambda x, y: map(lambda (x, y): x.extend(y), zip(x, y))
    extend_lists(statistics_map["weights_min"], min_wsmods)
    extend_lists(statistics_map["weights_max"], max_wsmods)
    extend_lists(statistics_map["biases_min"], min_bsmods)
    extend_lists(statistics_map["biases_max"], max_bsmods)

    iterations_sum += iterations
    statistics_map["iterations"] = iterations_sum
    statistics_map["last_etha"] = etha_per_iteration_calc[0][-1]

    # # Make a backup of the n'th neural network!
    # utils.save_pkl_file(nn, network_path+"nn_"+str(iterations_sum)+".pkl.gz")

    utils.save_pkl_file(nn, network_path+"nn.pkl.gz")
    utils.save_pkl_file(statistics_map, network_path+"statistics_map.pkl.gz")

def create_plots(network_name):
    network_path = "networks/"+network_name+"/"
    network_name = network_name[:20]+"..."

    nn = utils.load_pkl_file(network_path+"nn.pkl.gz")
    inp_tr = utils.load_pkl_file(network_path+"inputs_train.pkl.gz")
    targ_tr = utils.load_pkl_file(network_path+"targets_train.pkl.gz")
    inp_tst = utils.load_pkl_file(network_path+"inputs_test.pkl.gz")
    targ_tst = utils.load_pkl_file(network_path+"targets_test.pkl.gz")

    statistics_map = utils.load_pkl_file(network_path+"statistics_map.pkl.gz")
    errors_train = statistics_map["errors_train"]
    errors_test = statistics_map["errors_test"]
    errors_train_cecf = statistics_map["errors_train_cecf"]
    errors_test_cecf = statistics_map["errors_test_cecf"]
    etha_per_iteration = statistics_map["etha_per_iteration"]
    classes_train_true = statistics_map["classes_train_true"]
    classes_test_true = statistics_map["classes_test_true"]

    weights_min = statistics_map["weights_min"]
    weights_max = statistics_map["weights_max"]
    biases_min = statistics_map["biases_min"]
    biases_max = statistics_map["biases_max"]

    max_etha, min_etha = nn.get_max_min_etha()
    adapt_learn_pos, adapt_learn_neg = nn.get_adapt_speed()

    nl, bs, ws = nn.get_neuron_list(), nn.get_biases(), nn.get_weights()
    neural_name = nn.get_name_of_neural_network()
    hidden_function = nn.get_hidden_function()
    type_learning = nn.get_type_learning()
    network_type = nn.get_network_type()
    dpi_quality=500

    # def do_the_plots(directory, add_name, bs, ws, inputs_train, targets_train, inputs_test, targets_test, errors_train, errors_test):
    #     """  """
    directory = network_path
    add_name = ""
    network_properties = "Neural type: {}, Using hidden function: {}, Using type of learning: {}\n".format(neural_name, hidden_function, type_learning)+ \
                         "network name: {}, Neuron list: {}, max_etha: {}, min_etha: {}\n".format(network_name, nl, max_etha, min_etha)+ \
                         "adapt_learn_pos: {}, adapt_learn_neg: {}".format(adapt_learn_pos, adapt_learn_neg)

    legend_size = 8

    """ General plots """
    plt.clf()
    plt.figure()
    plt.title("Etha-Per-Iteration\n"+network_properties, fontsize=10)
    plt.plot(etha_per_iteration[1], etha_per_iteration[0], ".b")
    ax = plt.gca()
    ax.set_yscale("log", nonposy='clip')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Learning rate, log scale")
    plt.tight_layout()
    plt.savefig(directory+add_name+"etha_per_iterations.png", format="png", dpi=dpi_quality)

    plt.clf()
    plt.figure()
    plt.title("Error plot curve (RMSE)\n"+network_properties, fontsize=10)
    plt1, = plt.plot(np.arange(len(errors_train)), errors_train, "-b")
    plt2, = plt.plot(np.arange(len(errors_test)), errors_test, "-g")
    plt.xlabel("Iterations")
    plt.ylabel("Error (sqrt-mean-squared-error)")
    plt.legend([plt1, plt2], ["Train curve", "Test curve"], fontsize=legend_size)
    # plt.legend(fontsize=legend_size)
    plt.tight_layout()
    plt.savefig(directory+add_name+"sqrt_mean_squared_error.png", format="png", dpi=dpi_quality)

    plt.clf()
    plt.figure()
    plt.title("Error plot curve (CECF)\n"+network_properties, fontsize=10)
    plt1, = plt.plot(np.arange(len(errors_train_cecf)), errors_train_cecf, "-b")
    plt2, = plt.plot(np.arange(len(errors_test_cecf)), errors_test_cecf, "-g")
    plt.xlabel("Iterations")
    plt.ylabel("Error (sqrt-mean-squared-error)")
    ax = plt.gca()
    ax.set_yscale("log", nonposy='clip')
    plt.legend([plt1, plt2], ["Train curve", "Test curve"], fontsize=legend_size)
    # plt.legend(fontsize=legend_size)
    plt.tight_layout()
    plt.savefig(directory+add_name+"cross_entropy_cost_function_error.png", format="png", dpi=dpi_quality)

    """ Plots for max/min weights/biases """
    if len(nl) > 2:
        mult_factor = 255 / (len(nl)-2)
    default_color_max = "#FF0000"
    default_color_min = "#0000FF"
    ### max/min weights ###
    plts_max = []
    plts_min = []
    x = np.arange(0, len(weights_max[0]))

    plt.clf()
    plt.figure()
    plt.title("Weights Min/Max")
    plts_max.append(plt.plot(x, weights_max[0], "-", color=default_color_max)[0])
    plts_min.append(plt.plot(x, weights_min[0], "-", color=default_color_min)[0])
    for i, (w_max, w_min) in enumerate(zip(weights_max[1:], weights_min[1:])):
        color_max = (lambda x: x[:3]+(lambda x: "0"+x[2] if len(x) < 3 else x[2:])(hex(mult_factor*(i+1))).upper()+x[5:])(default_color_max)
        color_min = (lambda x: x[:3]+(lambda x: "0"+x[2] if len(x) < 3 else x[2:])(hex(mult_factor*(i+1))).upper()+x[5:])(default_color_min)
        plts_max.append(plt.plot(x, w_max, "-", color=color_max)[0])
        plts_min.append(plt.plot(x, w_min, "-", color=color_min)[0])
    plt.xlabel("Iterations")
    plt.ylabel("Max/Min values")
    plt.legend(plts_max+plts_min, ["Weights Max Layer 1"]+["Weights Max Layer {}".format(i+2) for i in xrange(len(nl)-2)]+
                                  ["Weights Min Layer 1"]+["Weights Min Layer {}".format(i+2) for i in xrange(len(nl)-2)], fontsize=legend_size)
    # plt.legend(fontsize=legend_size)
    plt.savefig(directory+add_name+"weights_max_min.png", format="png", dpi=dpi_quality)

    plts_max = []
    plts_min = []
    plt.clf()
    plt.figure()
    plt.title("Biases Min/Max")
    plts_max.append(plt.plot(x, biases_max[0], "-", color=default_color_max)[0])
    plts_min.append(plt.plot(x, biases_min[0], "-", color=default_color_min)[0])
    for i, (b_max, b_min) in enumerate(zip(biases_max[1:], biases_min[1:])):
        color_max = (lambda x: x[:3]+(lambda x: "0"+x[2] if len(x) < 3 else x[2:])(hex(mult_factor*(i+1))).upper()+x[5:])(default_color_max)
        color_min = (lambda x: x[:3]+(lambda x: "0"+x[2] if len(x) < 3 else x[2:])(hex(mult_factor*(i+1))).upper()+x[5:])(default_color_min)
        plts_max.append(plt.plot(x, b_max, "-", color=color_max)[0])
        plts_min.append(plt.plot(x, b_min, "-", color=color_min)[0])
    plt.xlabel("Iterations")
    plt.ylabel("Max/Min values")
    plt.legend(plts_max+plts_min, ["Biases Max Layer 1"]+["Biases Max Layer {}".format(i+2) for i in xrange(len(nl)-2)]+
                                  ["Biases Min Layer 1"]+["Biases Min Layer {}".format(i+2) for i in xrange(len(nl)-2)], fontsize=legend_size)
    # plt.legend(fontsize=legend_size)
    plt.savefig(directory+add_name+"biases_max_min.png", format="png", dpi=dpi_quality)

    # TODO: make finish for the regression and binary!
    if network_type == "classifier":
        """ Classifier plots """
        classes_train_false = 1 - np.array(classes_train_true).astype(np.float64) / inp_tr.shape[0]
        classes_test_false = 1 - np.array(classes_test_true).astype(np.float64) / inp_tst.shape[0]

        plt.clf()
        plt.figure()
        ax = plt.gca()
        plt.title("Prediction plot curve\n"+network_properties, fontsize=10)
        plt1, = plt.plot(np.arange(len(classes_train_false)), classes_train_false, "-b")
        plt2, = plt.plot(np.arange(len(classes_test_false)), classes_test_false, "-g")
        plt.xlabel("Iterations")
        plt.ylabel("Prediction (Failure rate)")
        plt.legend([plt1, plt2], ["Train curve", "Test curve"], fontsize=legend_size)
        # plt.legend(fontsize=legend_size)
        majorLocator = MultipleLocator(0.1)
        minorLocator = MultipleLocator(0.01)
        majorFormatter = FormatStrFormatter("%.2f")
        ax.set_ylim([0., 1.])
        ax.yaxis.set_major_locator(majorLocator)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_major_formatter(majorFormatter)

        plt.tight_layout()
        plt.savefig(directory+add_name+"predictions.png", format="png", dpi=dpi_quality)

        """ Confusion Matrix """
        results = nn.confusion_matrix_better_multiprocess([nl, bs, ws], inp_tr, targ_tr, inp_tst, targ_tst)
        confusion_tr = results[0]
        confusion_tst = results[1]

        # help from: http://stackoverflow.com/questions/2897826/confusion-matrix-with-number-of-classified-misclassified-instances-on-it-python
        conf_tr_norm = confusion_tr / (np.sum(1.*confusion_tr, axis=1)).reshape((10, 1))
        conf_tst_norm = confusion_tst / (np.sum(1.*confusion_tst, axis=1)).reshape((10, 1))

        x = -0.4
        y = 0.2

        plt.clf()
        plt.figure()
        fig = plt.gcf()
        ax = fig.add_subplot(111)

        res = ax.imshow(pylab.array(conf_tr_norm), cmap=plt.cm.jet, interpolation='nearest')
        for i, conf in enumerate(confusion_tr):
            for j, c in enumerate(conf):
                if c > 0:
                    plt.text(j+x, i+y, c, fontsize=12)
        plt.title("Confusion matrix of Train data")
        plt.xlabel("Real digit")
        plt.ylabel("Predicted digit")
        cb = fig.colorbar(res)
        plt.savefig(network_path+"confusion_matrix_train.png", format="png")

        # fig = plt.figure()
        plt.clf()
        fig = plt.gcf()
        ax = fig.add_subplot(111)

        res = ax.imshow(pylab.array(conf_tst_norm), cmap=plt.cm.jet, interpolation='nearest')
        for i, conf in enumerate(confusion_tst):
            for j, c in enumerate(conf):
                if c>0:
                    plt.text(j+x, i+y, c, fontsize=14)
        plt.title("Confusion matrix of Test data")
        plt.xlabel("Real digit")
        plt.ylabel("Predicted digit")
        cb = fig.colorbar(res)
        plt.savefig(network_path+"confusion_matrix_test.png", format="png")

        print("confusion_tr =\n{}".format(confusion_tr))
        print("confusion_tst =\n{}".format(confusion_tst))
    elif network_type == "binary":
        pass
    elif network_type == "regression":
        pass

def create_sqrt_plot(directory, add_name, network_name, errors_train, errors_test, nl, neural_type, hidden_function, type_learning, with_sgd, with_adaptive, with_no_random):
    plt.figure()
    plt.title("Error plot curve\nNeural type: {}, Using hidden function: {}, Using type of learning: {}\n".format(neural_type, hidden_function, type_learning)+
              "Network name: {}, Neuron list: {}\n".format(network_name, nl)+
              "Learning Mode: {}, With Adaptive? {}, Random Allowed? {}".format(("SGD" if with_sgd else "BGD"), (("YES" if with_adaptive else "NO")), ("NO" if with_no_random else "YES")), fontsize=10)
    plt1, = plt.plot(np.arange(len(errors_train)), errors_train, "-b")
    plt2, = plt.plot(np.arange(len(errors_test)), errors_test, "-g")
    plt.xlabel("Iterations")
    plt.ylabel("Error (sqrt-mean-squared-error)")
    plt.legend([plt1, plt2], ["Train curve", "Test curve"])
    plt.ylim([np.max([np.min([np.min(errors_train), np.min(errors_test)])-0.1, 0.]),
              np.max([np.max(errors_train), np.max(errors_test)])+0.1])
    plt.tight_layout()
    plt.savefig(directory+add_name+"sqrt_mean_squared_error.png", format="png", dpi=500)

def create_plots_binadder(network_name, with_sgd, with_adaptive, with_no_random):
    network_path = "networks/"+network_name+"/"

    utils.check_create_dir(network_path+"merged_plots_train/")
    utils.check_create_dir(network_path+"merged_plots_test/")
    utils.check_create_dir(network_path+"histogram_bits_train/")
    utils.check_create_dir(network_path+"histogram_bits_test/")

    nn = utils.load_pkl_file(network_path+"nn.pkl.gz")
    bs, ws = nn.get_biases(), nn.get_weights()
    inp_tr = utils.load_pkl_file(network_path+"inputs_train.pkl.gz")
    targ_tr = utils.load_pkl_file(network_path+"targets_train.pkl.gz")
    inp_tst = utils.load_pkl_file(network_path+"inputs_test.pkl.gz")
    targ_tst = utils.load_pkl_file(network_path+"targets_test.pkl.gz")

    statistics_map = utils.load_pkl_file(network_path+"statistics_map.pkl.gz")
    errors_train = statistics_map["errors_train"]
    errors_test = statistics_map["errors_test"]
    errors_train_cecf = statistics_map["errors_train_cecf"]
    errors_test_cecf = statistics_map["errors_test_cecf"]
    etha_per_iteration = statistics_map["etha_per_iteration"]
    iterations_sum = statistics_map["iterations"]
    # last_etha = statistics_map["last_etha"]

    nl = nn.get_neuron_list()
    neural_type = nn.get_name_of_neural_network()
    hidden_function = nn.get_hidden_function()
    type_learning = nn.get_type_learning()

    # def do_the_plots(directory, add_name, bs, ws, inputs_train, targets_train, inputs_test, targets_test, errors_train, errors_test):
    #     """  """
    directory = network_path
    add_name = ""
    """ General plots """
    plt.figure()
    plt.title("Etha-Per-Iteration\nNeural type: {}, Using hidden function: {}, Using type of learning: {}\n".format(neural_type, hidden_function, type_learning)+
              "Network name: {}, Neuron list: {}\n".format(network_name, nl)+
              "Learning Mode: {}, With Adaptive? {}, Random Allowed? {}".format(("SGD" if with_sgd else "BGD"), (("YES" if with_adaptive else "NO")), ("NO" if with_no_random else "YES")), fontsize=10)
    plt.plot(etha_per_iteration[1], etha_per_iteration[0], ".b")
    ax = plt.gca()
    ax.set_yscale("log", nonposy='clip')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Learning rate, log scale")
    plt.tight_layout()
    plt.savefig(directory+add_name+"etha_per_iterations.png", format="png", dpi=500)

    plt.figure()
    plt.title("Error plot curve\nNeural type: {}, Using hidden function: {}, Using type of learning: {}\n".format(neural_type, hidden_function, type_learning)+
              "Network name: {}, Neuron list: {}\n".format(network_name, nl)+
              "Learning Mode: {}, With Adaptive? {}, Random Allowed? {}".format(("SGD" if with_sgd else "BGD"), (("YES" if with_adaptive else "NO")), ("NO" if with_no_random else "YES")), fontsize=10)
    plt1, = plt.plot(np.arange(len(errors_train)), errors_train, "-b")
    plt2, = plt.plot(np.arange(len(errors_test)), errors_test, "-g")
    plt.xlabel("Iterations")
    plt.ylabel("Error (sqrt-mean-squared-error)")
    plt.legend([plt1, plt2], ["Train curve", "Test curve"])
    plt.ylim([np.max([np.min([np.min(errors_train), np.min(errors_test)])-0.1, 0.]),
              np.max([np.max(errors_train), np.max(errors_test)])+0.1])
    plt.tight_layout()
    plt.savefig(directory+add_name+"sqrt_mean_squared_error.png", format="png", dpi=500)

    plt.figure()
    plt.title("Error plot curve (cross-entropy-cost-function)\nNeural type: {}, Using hidden function: {}, Using type of learning: {}\n".format(neural_type, hidden_function, type_learning)+
              "Network name: {}, Neuron list: {}\n".format(network_name, nl)+
              "Learning Mode: {}, With Adaptive? {}, Random Allowed? {}".format(("SGD" if with_sgd else "BGD"), (("YES" if with_adaptive else "NO")), ("NO" if with_no_random else "YES")), fontsize=10)
    plt1, = plt.plot(np.arange(len(errors_train_cecf)), errors_train_cecf, "-b")
    plt2, = plt.plot(np.arange(len(errors_test_cecf)), errors_test_cecf, "-g")
    plt.xlabel("Iterations")
    plt.ylabel("Error (cross-entropy-cost-function)")
    plt.legend([plt1, plt2], ["Train curve", "Test curve"])
    plt.ylim([np.max([np.min([np.min(errors_train), np.min(errors_test)])-0.05, 0.]),
              np.max([np.max(errors_train_cecf), np.max(errors_test_cecf)])+0.05])
    plt.tight_layout()
    plt.savefig(directory+add_name+"cross_entropy_cost_function.png", format="png", dpi=500)

    """ For Train set """
    out_tr = nn.calculate_forward_many(inp_tr, bs, ws)
    mse_train_separate = nn.mean_square_error_separate(out_tr, targ_tr)
    mse_train_separate_x = np.arange(mse_train_separate.shape[0])
    width = 1./10/2
    bins_train = np.arange(0, 1.+width, width)
    bins_plot_x_train = bins_train[:-1]+width*.5
    histogram_train = np.histogram(mse_train_separate, bins=bins_train)[0]
    sorted_mse_train_x = np.arange(mse_train_separate.shape[0])
    sorted_mse_train_y = np.sort(mse_train_separate)
    mse_train_first_derivative_x = sorted_mse_train_x[:-1]+0.5
    mse_train_first_derivative_y = np.array([x2-x1 for x1, x2 in zip(sorted_mse_train_y[:-1], sorted_mse_train_y[1:])])

    """ For Test set """
    out_tst = nn.calculate_forward_many(inp_tst, bs, ws)
    mse_test_separate = nn.mean_square_error_separate(out_tst, targ_tst)
    mse_test_separate_x = np.arange(mse_train_separate.shape[0])
    width = 1./10/2
    bins_test = np.arange(0, 1.+width, width)
    bins_plot_x_test = bins_test[:-1]+width*.5
    histogram_test = np.histogram(mse_test_separate, bins=bins_train)[0]
    sorted_mse_test_x = np.arange(mse_test_separate.shape[0])
    sorted_mse_test_y = np.sort(mse_test_separate)
    mse_test_first_derivative_x = sorted_mse_test_x[:-1]+0.5
    mse_test_first_derivative_y = np.array([x2-x1 for x1, x2 in zip(sorted_mse_test_y[:-1], sorted_mse_test_y[1:])])

    """ Create a histogramm of correct binarys calculated """
    round_up = lambda x: 0. if x < 0.5 else 1.
    round_up_vec = np.vectorize(round_up)
    out_tr_round = round_up_vec(out_tr)
    out_tst_round = round_up_vec(out_tst)

    correct_tr_bits = np.sum(targ_tr==out_tr_round, axis=1)
    correct_tst_bits = np.sum(targ_tst==out_tst_round, axis=1)

    bins = [i-0.5 for i in xrange(0, nl[-1]+2)]
    print("bins = {}".format(bins))

    plt.figure()
    fig = plt.gcf()
    ax = plt.gca()
    plt.xticks([i for i in xrange(0, nl[-1]+1)])
    plt.xlim([-0.5, nl[-1]+0.5])
    plt.ylim([0, inp_tr.shape[0]])
    plt.hist(correct_tr_bits, bins=bins)
    plt.title("Histogram for Train data")
    plt.xlabel("Correct calculated bits")
    plt.ylabel("Amount of correct bits")
    plt.savefig(directory+add_name+"histogram_bits_train/histogram_bits_train_iter_{}.png".format(iterations_sum), format="png", dpi=500)

    plt.figure()
    fig = plt.gcf()
    ax = plt.gca()
    plt.xticks([i for i in xrange(0, nl[-1]+1)])
    plt.xlim([-0.5, nl[-1]+0.5])
    plt.ylim([0, inp_tst.shape[0]])
    plt.hist(correct_tst_bits, bins=bins)
    plt.title("Histogram for Test data")
    plt.xlabel("Correct calculated bits")
    plt.ylabel("Amount of correct bits")
    plt.savefig(directory+add_name+"histogram_bits_test/histogram_bits_test_iter_{}.png".format(iterations_sum), format="png", dpi=500)

def check_calculation_correctness(network_name):
    network_path = "networks/"+network_name+"/"

    nn = utils.load_pkl_file(network_path+"nn.pkl.gz")
    inp_tr = utils.load_pkl_file(network_path+"inputs_train.pkl.gz")
    targ_tr = utils.load_pkl_file(network_path+"targets_train.pkl.gz")
    inp_tst = utils.load_pkl_file(network_path+"inputs_test.pkl.gz")
    targ_tst = utils.load_pkl_file(network_path+"targets_test.pkl.gz")

    bs, ws = nn.get_biases(), nn.get_weights()
    out_tr = nn.calculate_forward_many(inp_tr, bs, ws)
    out_tst = nn.calculate_forward_many(inp_tst, bs, ws)

    # round_up = lambda x: 0.1 if x < 0.5 else 0.9
    round_up = lambda x: 0. if x < 0.5 else 1.
    round_up_vec = np.vectorize(round_up)
    out_tr_round = round_up_vec(out_tr)
    out_tst_round = round_up_vec(out_tst)
    pos_tr = np.sum(np.all(out_tr_round==targ_tr, axis=1))
    pos_tst = np.sum(np.all(out_tst_round==targ_tst, axis=1))
    neg_tr = inp_tr.shape[0] - pos_tr
    neg_tst = inp_tst.shape[0] - pos_tst

    print("pos_tr = {}".format(pos_tr))
    print("neg_tr = {}".format(neg_tr))
    print("pos_tst = {}".format(pos_tst))
    print("neg_tst = {}".format(neg_tst))

    with open(network_path+"statistics_calculations.txt", "w") as fout:
        fout.write("Statistics for the network: {}\n".format(network_name))
        fout.write("For Train:\n")
        fout.write("Correct calculated: {:8}   False calculated: {:8}\n".format(pos_tr, neg_tr))
        fout.write("For Test:\n")
        fout.write("Correct calculated: {:8}   False calculated: {:8}\n".format(pos_tst, neg_tst))

def create_confusion_matrix(network_name):
    network_path = "networks/"+network_name+"/"

    nn = utils.load_pkl_file(network_path+"nn.pkl.gz")
    inp_tr = utils.load_pkl_file(network_path+"inputs_train.pkl.gz")
    targ_tr = utils.load_pkl_file(network_path+"targets_train.pkl.gz")
    inp_tst = utils.load_pkl_file(network_path+"inputs_test.pkl.gz")
    targ_tst = utils.load_pkl_file(network_path+"targets_test.pkl.gz")

    nl, bs, ws = nn.get_neuron_list(), nn.get_biases(), nn.get_weights()

    results = nn.confusion_matrix_better_multiprocess([nl, bs, ws], inp_tr, targ_tr, inp_tst, targ_tst)
    confusion_tr = results[0]
    confusion_tst = results[1]

    # help from: http://stackoverflow.com/questions/2897826/confusion-matrix-with-number-of-classified-misclassified-instances-on-it-python
    conf_tr_norm = confusion_tr / (np.sum(1.*confusion_tr, axis=1)).reshape((10, 1))
    conf_tst_norm = confusion_tst / (np.sum(1.*confusion_tst, axis=1)).reshape((10, 1))

    x = -0.4
    y = 0.2

    fig = plt.figure()
    ax = fig.add_subplot(111)

    res = ax.imshow(pylab.array(conf_tr_norm), cmap=plt.cm.jet, interpolation='nearest')
    for i, conf in enumerate(confusion_tr):
        for j, c in enumerate(conf):
            if c > 0:
                plt.text(j+x, i+y, c, fontsize=12)
    plt.title("Confusion matrix of Train data")
    plt.xlabel("Real digit")
    plt.ylabel("Predicted digit")
    cb = fig.colorbar(res)
    plt.savefig(network_path+"confusion_matrix_train.png", format="png")

    fig = plt.figure()
    ax = fig.add_subplot(111)

    res = ax.imshow(pylab.array(conf_tst_norm), cmap=plt.cm.jet, interpolation='nearest')
    for i, conf in enumerate(confusion_tst):
        for j, c in enumerate(conf):
            if c>0:
                plt.text(j+x, i+y, c, fontsize=14)
    plt.title("Confusion matrix of Test data")
    plt.xlabel("Real digit")
    plt.ylabel("Predicted digit")
    cb = fig.colorbar(res)
    plt.savefig(network_path+"confusion_matrix_test.png", format="png")

    print("confusion_tr =\n{}".format(confusion_tr))
    print("confusion_tst =\n{}".format(confusion_tst))

def do_create_and_learn():
    bits = 5
    create_binary_adder_neural_network(bits=bits, name_prefix="sgd_sigmoid_morehidden_")
    network_name = "sgd_sigmoid_morehidden_neural_network_binary_adder_"+str(bits)+"_bits"

    # create_random_network("random1", 10, 10)
    # network_name = "random1"

    learn_neural_network(network_name, 20000)
    create_plots(network_name)
    check_calculation_correctness(network_name)

def learn_digit_recognizer():
    network_name = "digit_recognizer_tr_1000_tst_1000"
    create_digit_recognizer_neural_network(network_name, 1000, 1000)
    learn_neural_network(network_name, 1000)
    create_plots(network_name)

def main(argv):
    if len(argv) >= 2:
        if argv[1] == "digitmany":
            print("it is digitmany!")

            # best parameters: sgd, sidmoid, adaptiv
            # random: max_etha=0.1, min_etha=0.0001
            # adaptiv: pos=1.02, neg=0.75
            train_amount = 1000
            test_amount = 200

            index = 2
            name_suffix = argv[index]; index += 1
            iterations = int(argv[index]); index += 1
            is_cont = bool(int(argv[index])); index += 1
            def printlocal(x, dic):
                print("{} = {}".format(x, dic[x]))
            printlocal("name_suffix", locals())
            printlocal("iterations", locals())
            printlocal("is_cont", locals())
            network_name = PREFIX_DIGIT_RECOGNIZER+"_"+name_suffix

            if is_cont == False:
                neural_list = eval(argv[index]); index += 1
                max_etha = float(argv[index]); index += 1
                min_etha = float(argv[index]); index += 1
                if max_etha < min_etha:
                    max_etha, min_etha = min_etha, max_etha
                adapt_learn_pos = float(argv[index]); index += 1
                adapt_learn_neg = float(argv[index]); index += 1

                # TODO make a clean input in Terminal possible!!!
                hidden_function = argv[index]; index += 1
                type_learning = argv[index]; index += 1
                is_adaptive = bool(int(argv[index])); index += 1
                is_no_random = bool(int(argv[index])); index += 1

                create_digit_recognizer_neural_network(network_name, neural_list=neural_list, train_amount=train_amount, test_amount=test_amount,
                                                       max_etha=max_etha, min_etha=min_etha, adapt_learn_pos=adapt_learn_pos,
                                                       adapt_learn_neg=adapt_learn_neg, hidden_function=hidden_function,
                                                       type_learning=type_learning, is_adaptive=is_adaptive, is_no_random=is_no_random)
            for _ in xrange(iterations):
                learn_neural_network(network_name, iterations=1000)
                create_plots(network_name)
                # create_confusion_matrix(network_name)

        elif argv[1] == "addermany":
            if len(argv) != 12:
                print("Not enough arguments, needed 12!")
                sys.exit(2)
            command = argv[1]
            bits = int(argv[2])
            name_suffix = argv[3]
            iterations = int(argv[4])
            is_cont = bool(int(argv[5]))
            with_adaptive = bool(int(argv[6]))
            with_no_random = bool(int(argv[7]))
            first_etha = float(argv[8])
            random_count = int(argv[9])
            type_learning = argv[10]
            hidden_function = argv[11]

            print("command = {}".format(command))
            print("bits = {}".format(bits))
            print("name_suffix = {}".format(name_suffix))
            print("iterations = {}".format(iterations))
            print("is_cont = {}".format(is_cont))
            print("with_adaptive = {}".format(with_adaptive))
            print("with_no_random = {}".format(with_no_random))
            print("first_etha = {}".format(first_etha))
            print("random_count = {}".format(random_count))
            print("type_learning = {}".format(type_learning))
            print("hidden_function = {}".format(hidden_function))

            if (type_learning != "sgd") and (type_learning != "bgd"):
                print("type_learning can be: sgd, bgd")
                sys.exit(2)
            if (hidden_function != "sigmoid") and (hidden_function != "tanh"):
                print("hidden_function can be: sigmoid, tanh")
                sys.exit(2)
            with_sgd = True if type_learning == "sgd" else False
            # type_learning = "sgd" if with_sgd else "bgd"
            # hidden_function = "sigmoid"
            # hidden_function = "tanh"

            network_name = "binary_adder_{}_bits".format(bits)+"_"+name_suffix

            if not is_cont:
                create_binary_adder_neural_network(network_name, bits=bits, max_etha=1., min_etha=0.001, type_learning=type_learning, hidden_function=hidden_function, first_etha=first_etha)
            create_plots_binadder(network_name, with_sgd, with_adaptive, with_no_random)
            for _ in xrange(iterations):
                learn_neural_network(network_name, iterations=100, with_sgd=with_sgd, with_adaptive=with_adaptive, with_no_random=with_no_random)
                create_plots_binadder(network_name, with_sgd, with_adaptive, with_no_random)
            check_calculation_correctness(network_name)
        elif argv[1] == "create4types":
            if len(argv) <= 2:
                print("More than 2 arguments is needed!")
                sys.exit(2)
            if argv[2] == "binadder":
                if len(argv) != 8:
                    print("8 arguments needed!")
                    sys.exit(2)
                bits = int(argv[3])
                name1 = argv[4]
                name2 = argv[5]
                name3 = argv[6]
                name4 = argv[7]

                network_name1 = "binary_adder_{}_bits_{}".format(bits, name1)
                network_name2 = "binary_adder_{}_bits_{}".format(bits, name2)
                network_name3 = "binary_adder_{}_bits_{}".format(bits, name3)
                network_name4 = "binary_adder_{}_bits_{}".format(bits, name4)
                arguments = {"bits": bits, "max_etha": 1., "min_etha": 0.001, "type_learning": "bgd", "hidden_function": "sigmoid", "first_etha": 0.005}
                # create_binary_adder_neural_network(network_name1, bits=bits, max_etha=1., min_etha=0.001, type_learning="bgd", hidden_function="sigmoid", first_etha=0.005)

                arguments.update({"is_sgd": False, "is_adaptive": False, "is_no_random": False})
                create_binary_adder_neural_network(network_name1, **arguments)
                arguments.update({"is_sgd": False, "is_adaptive": True, "is_no_random": False})
                create_binary_adder_neural_network(network_name2, **arguments)
                arguments.update({"is_sgd": False, "is_adaptive": False, "is_no_random": True})
                create_binary_adder_neural_network(network_name3, **arguments)
                arguments.update({"is_sgd": False, "is_adaptive": True, "is_no_random": True})
                create_binary_adder_neural_network(network_name4, **arguments)

                full_name1 = "networks/" + network_name1
                full_name2 = "networks/" + network_name2
                full_name3 = "networks/" + network_name3
                full_name4 = "networks/" + network_name4

                # TODO: COPY ONLY stats, inputs and targets!!!
                src_files = os.listdir(full_name1)
                for file_name in src_files:
                    if file_name == "nn.pkl.gz" or file_name == "nn_init.pkl.gz":
                        print("file '{}' will not be copied!".format(file_name))
                        continue
                    full_file_name = full_name1+"/"+file_name
                    if (os.path.isfile(full_file_name)):
                        shutil.copy(full_file_name, full_name2+"/"+file_name)
                        shutil.copy(full_file_name, full_name3+"/"+file_name)
                        shutil.copy(full_file_name, full_name4+"/"+file_name)
                        # print("copied file '{}' to folder(s) '{}'".format(file_name, [full_name2, full_name3, full_name4]))
        elif argv[1] == "copystats":
            if len(argv) <= 2:
                print("More than 2 arguments is needed!")
                sys.exit(2)
            if argv[2] == "binadder":
                if len(argv) != 8:
                    print("8 arguments needed!")
                    sys.exit(2)
                bits = int(argv[3])
                name1 = argv[4]
                name2 = argv[5]
                name3 = argv[6]
                name4 = argv[7]

                network_name1 = "binary_adder_{}_bits_{}".format(bits, name1)
                network_name2 = "binary_adder_{}_bits_{}".format(bits, name2)
                network_name3 = "binary_adder_{}_bits_{}".format(bits, name3)
                network_name4 = "binary_adder_{}_bits_{}".format(bits, name4)

                full_name1 = "networks/" + network_name1
                full_name2 = "networks/" + network_name2
                full_name3 = "networks/" + network_name3
                full_name4 = "networks/" + network_name4

                utils.check_create_dir("networks/statistics")
                shutil.copy(full_name1+"/statistics_map.pkl.gz", "networks/statistics/statistics_map_"+name1+".pkl.gz")
                shutil.copy(full_name2+"/statistics_map.pkl.gz", "networks/statistics/statistics_map_"+name2+".pkl.gz")
                shutil.copy(full_name3+"/statistics_map.pkl.gz", "networks/statistics/statistics_map_"+name3+".pkl.gz")
                shutil.copy(full_name4+"/statistics_map.pkl.gz", "networks/statistics/statistics_map_"+name4+".pkl.gz")

                utils.check_create_dir("networks/sqrt_mean_squared_error")
                shutil.copy(full_name4+"/sqrt_mean_squared_error.png", "networks/sqrt_mean_squared_error/sqrt_mean_squared_error_"+name1+"png")
                shutil.copy(full_name4+"/sqrt_mean_squared_error.png", "networks/sqrt_mean_squared_error/sqrt_mean_squared_error_"+name2+"png")
                shutil.copy(full_name4+"/sqrt_mean_squared_error.png", "networks/sqrt_mean_squared_error/sqrt_mean_squared_error_"+name3+"png")
                shutil.copy(full_name4+"/sqrt_mean_squared_error.png", "networks/sqrt_mean_squared_error/sqrt_mean_squared_error_"+name4+"png")
    elif argv[1] == "-h" or argv[1] == "--help":
            print("usages:\n"
                  "for digit recognition:\n"
                  "python2 create_network_folder.py digit [train_amount=1000 test_amount=1000 [name_suffix=\"\" [iterations=10 [bgd/sgd [cont]]]]]\n"
                  "for binary adder:\n"
                  "python2 create_network_folder.py adder [bits=4 [name_suffix=\"\" [iterations=15 [cont=False]]]]]\n"
                  "python2 create_network_folder.py addermany <bits> <name_suffix> <iters> <cont> <adaptive> <no_random> <etha> <random_networks>")
    else:
        print("No args!")

if __name__ == "__main__":
    main(sys.argv)
