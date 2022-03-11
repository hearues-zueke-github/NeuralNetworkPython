#! /bin/bash

remote_folder=/calc/students/ziko/nn_learning/neuronal_network/networks/mnist_networks
host_folder=/home/doublepmcl/Documents/saved_networks_remote/mnist_networks

mkdir -p $host_folder

# folder=data_size_7x7_nl_49_100_10_with_momentum_1_degree_True_func_tanh
# scp -rp figi150:$remote_folder/$folder $host_folder

folder=network_data_size_28x28_nl_784_1000_10_with_momentum_1_degree_True_func_tanh
scp -rp figi150:$remote_folder/$folder $host_folder

# folder=network_data_size_28x28_nl_784_1200_10_with_momentum_1_degree_True_func_tanh
# scp -rp figi150:$remote_folder/$folder $host_folder


# remote_folder=/calc/students/ziko/nn_learning/neuronal_network/networks_gradient_descent
# host_folder=/home/doublepmcl/Documents/networks_gradient_descent_remote

# folder=try_nr_1
# scp -rp figi150:$remote_folder/$folder/pictures $host_folder/$folder

# folder=try_nr_2
# scp -rp figi150:$remote_folder/$folder/pictures $host_folder/$folder


remote_folder=/calc/students/ziko/nn_learning/neuronal_network/network_matrix_multiply
host_folder=/home/doublepmcl/Documents/saved_networks_remote/network_matrix_multiply

# folder=inp_5_out_3_nl_5_65_65_3_with_momentum_1_degree_True_func_tanh
# scp -rp figi150:$remote_folder/$folder $host_folder

#folder=inp_8_out_5_nl_8_150_150_5_with_momentum_1_degree_True_func_tanh
#scp -rp figi150:$remote_folder/$folder $host_folder
