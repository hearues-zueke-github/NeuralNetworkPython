#! /bin/bash

host_folder=/home/doublepmcl/git/NeuralNetworkPython/NeuralNetworkNetbeansJava/src/main/python/better_network
remote_folder=/calc/students/ziko/nn_learning/neuronal_network/

host_folder_utils=/home/doublepmcl/git/NeuralNetworkPython/NeuralNetworkNetbeansJava/src/main/python/utils
remote_folder_utils=/calc/students/ziko/nn_learning/utils/

scp \
$host_folder/neuralnetwork.py \
$host_folder/multi_layer_network_binary_classifier.py \
$host_folder/binary_matrix_data_set.py \
$host_folder/gradient_descent.py \
$host_folder/mnist.py \
$host_folder/mnist_pictures.py \
figi150:$remote_folder

scp \
$host_folder_utils/MathUtils.py \
$host_folder_utils/Utils.py \
figi150:$remote_folder_utils
