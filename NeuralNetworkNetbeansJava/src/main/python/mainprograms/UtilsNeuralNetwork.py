#! /usr/bin/python2.7

from __init__ import *

print_variable = lambda variable_name: utils.print_variable(variable_name, globals())

def save_new_random_network(neural_list, file_path):
    nn = NeuralNetwork()
    nn.set_neuron_list(neural_list)
    nn.init_random_weights()

    tn = TrainedNetwork()
    tn.set_file_path(file_path)
    tn.set_network(nn)
    tn.save_network()
# def save_new_random_network

def create_new_random_network(neural_list, additional_names="", is_autoencoder=False):
    name_part = "".join(map(lambda x: "_"+str(x), neural_list))
    
    file_path = "lnn"+additional_names+("_autoencoder" if is_autoencoder else "")+name_part+".pkl.gz"

    utils.print_variable("neural_list", locals())
    utils.print_variable("file_path", locals())

    save_new_random_network(neural_list, file_path)
# def create_new_random_network

def create_normal_networks():
    create_new_random_network([14*14, 14*2, 10], additional_names="_14x14")
    create_new_random_network([14*14, 14*4, 10], additional_names="_14x14")
    create_new_random_network([14*14, 14*4, 14*2, 10], additional_names="_14x14")

def create_autoencoder_networks():
    create_new_random_network([14*14, 14*2, 10, 14*2, 14*14], additional_names="_14x14", is_autoencoder=True)
    create_new_random_network([14*14, 14*4, 10, 14*4, 14*14], additional_names="_14x14", is_autoencoder=True)
    create_new_random_network([14*14, 14*4, 14*2, 10, 14*2, 14*4, 14*14], additional_names="_14x14", is_autoencoder=True)

def create_binary_adder_network_inputs_targets(binary_adder_sizes):
    # binary_adder_sizes = [2, 3, 4, 5, 6, 8]
    list_of_neural_list = [[x*2, x*3, x+1] for x in binary_adder_sizes]
    print("list_of_neural_list = "+str(list_of_neural_list))

    for neural_list in list_of_neural_list:
        create_new_random_network(neural_list, additional_names="_binadder")

    for i in binary_adder_sizes:
        inputs_targets = binadd.get_binaryadder_inputs_targets(i)
        file_path = "inputs_targets_binary_"+str(i)+"_bits.pkl.gz"

        utils.save_pkl_file(inputs_targets, file_path)

def change_weights_of_network(file_path):
    tn = TrainedNetwork()
    tn.set_file_path(file_path)
    tn.load_network()
    nn = tn.nn
    print("Loaded network")
    nn.init_random_weights()
    print("Init random weights")
    tn.save_network()
    print("Save network")
# def change_weights_of_network
