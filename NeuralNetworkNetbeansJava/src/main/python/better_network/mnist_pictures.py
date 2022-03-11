#! /usr/bin/python2.7

import sys
import cPickle
import gzip
import numpy as np
from copy import deepcopy

from PIL import Image

from multiprocessing import Process

rnd = np.random.random

def save_X_as_img(X, size, amount_horizontal, file_name):
    assert X.shape[0] % amount_horizontal == 0
    
    x = amount_horizontal
    y = X.shape[0] // x
    pix = np.zeros((size*y, size*x, 3)).astype(np.uint8)
    
    pix[:, :] = (X*255). \
            astype(np.uint8). \
            reshape((-1, size, size)). \
            transpose(0, 2, 1). \
            reshape((-1, x*size, size)). \
            transpose(0, 2, 1). \
            reshape((-1, x*size, 1))

    img = Image.fromarray(pix)
    img.save(file_name, "PNG")

def create_mnist_with_noise(size):
    data = np.load("mnist_{}x{}.npz.gz".format(size, size))
    X_tr = data["X_train"] # [:5000]
    T_tr = data["T_train"] # [:5000]
    V_tr = data["V_train"] # [:5000]
    print("Loaded MNIST: size: {}, X_tr.shape: {}".format(size, X_tr.shape))

    print("Save X_tr_{}x{} as image".format(size, size))
    save_X_as_img(X_tr, size, 100, "mnist_{}x{}.png".format(size, size))

    X_tr_noise = X_tr+(rnd(X_tr.shape)*2-1)*0.05
    X_tr_noise[X_tr_noise<0.] = 0.
    X_tr_noise[X_tr_noise>1.] = 1.

    print("Save X_tr_{}x{}_noise as image".format(size, size))
    save_X_as_img(X_tr_noise, size, 100, "mnist_{}x{}_noise.png".format(size, size))

    X_tr_comb = np.vstack((X_tr, X_tr_noise))
    T_tr = data["T_train"]
    T_tr_comb = np.vstack((T_tr, T_tr))
    V_tr = data["V_train"]
    V_tr_comb = np.hstack((V_tr, V_tr))

    idx = np.random.permutation(np.arange(0, X_tr_comb.shape[0]))
    X_tr_comb = X_tr_comb[idx]
    T_tr_comb = T_tr_comb[idx]
    V_tr_comb = V_tr_comb[idx]

    print("Save X_tr_{}x{}_comb as image".format(size, size))
    save_X_as_img(X_tr_comb, size, 200, "mnist_{}x{}_comb.png".format(size, size))

    print("Saving MNIST {}x{} with random noise as mnist_{}x{}_tr_noise.npz.gz".format(size, size, size, size))
    with open("mnist_{}x{}_tr_noise.npz.gz".format(size, size), "wb") as f:
        np.savez_compressed(f, X_train=X_tr_comb, T_train=T_tr_comb, V_train=V_tr_comb,
                               X_valid=data["X_valid"], T_valid=data["T_valid"], V_valid=data["V_valid"],
                               X_test=data["X_test"], T_test=data["T_test"], V_test=data["V_test"])

if __name__ == "__main__":
    sizes = [7, 14, 28]
    procs = [Process(target=create_mnist_with_noise, args=(size, )) for size in sizes]
    # for size in sizes:
    #     create_mnist_with_noise(size)

    for proc in procs: proc.start()
    for proc in procs: proc.join()
