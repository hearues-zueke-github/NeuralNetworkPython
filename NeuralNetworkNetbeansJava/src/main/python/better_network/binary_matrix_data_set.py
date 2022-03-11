#! /usr/bin/python2.7

import os
import socket

import numpy as np

f_sig = lambda X: 1 / (1+np.exp(-X))
get_file_name = lambda m, k, n: "matrix_multiply_data_m_{}_k_{}_n_{}.npz".format(m, k, n)
get_file_name_rounded = lambda m, k, n: "matrix_multiply_data_m_{}_k_{}_n_{}_rounded.npz".format(m, k, n)

def get_new_X_T(m, k, n):
    X = np.random.random((m, k))*2.-1.
    
    T = X
    T += np.random.random(T.shape)*0.01
    k1 = k
    k2 = np.random.randint(2, 10)
    for _ in xrange(0, 3):
        A = np.random.random((k1, k2))*2.-1.
        T = np.tanh(np.dot(T, A))
        k1 = k2
        k2 = np.random.randint(2, 10)

    A = np.random.random((k1, n))*2.-1.
    T = np.tanh(np.dot(T, A))
    T += np.random.random(T.shape)*0.01
    
    return X, T

def get_rounded_matrix(X, threshold):
    X_round = np.zeros(X.shape)
    X_round[X >= threshold] = 1.
    # X_round[X < threshold] = 0.

    return X_round

def save_file_data(file_path_data_set, X, T):
    length = X.shape[0]
    idx_1 = int(length*0.6)
    idx_2 = int(length*0.8)
    X_train = X[:idx_1]
    T_train = T[:idx_1]
    X_valid = X[idx_1:idx_2]
    T_valid = T[idx_1:idx_2]
    X_test = X[idx_2:]
    T_test = T[idx_2:]

    with open(file_path_data_set, "wb") as f:
        np.savez(f, X_train=X_train, T_train=T_train,
                    X_valid=X_valid, T_valid=T_valid,
                    X_test=X_test, T_test=T_test)


def get_file_path_data_set(m, k, n):
    hostname = socket.gethostname()
    if "figi" in hostname:
        folder_path = "/calc/students/ziko/nn_learning/neuronal_network/matrix_multiply_data"
    else:
        folder_path = os.path.expanduser("~")+"/Documents/matrix_multiply_data"

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_path_data_set = folder_path+"/"+get_file_name_rounded(m, k, n)

    if not os.path.exists(file_path_data_set):
        X, T = get_new_X_T(m, k, n)
        T_round = get_rounded_matrix(T, 0.)
        save_file_data(file_path_data_set, X, T_round)

    return file_path_data_set

if __name__ == "__main__":
    k = 3
    for n in xrange(3, 9):
        print("data_size: n: {}".format(n))
        X, T, T_round = get_new_X_T(5000, k, n)

        file_name = get_file_name(n, k)
        save_file_data(file_name, X, T)

        file_name = get_file_name_rounded(n, k)
        save_file_data(file_name, X, T_round)
