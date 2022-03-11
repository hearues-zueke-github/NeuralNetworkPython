#! /usr/bin/python3

import numpy as np
from PIL import Image

mnist = np.load("save_mnist_matrices.npz")

inp_tr = mnist["inp_tr"]
inp_vld = mnist["inp_vld"]
inp_tst = mnist["inp_tst"]

targ_tr = mnist["targ_tr"]
targ_vld = mnist["targ_vld"]
targ_tst = mnist["targ_tst"]

def get_img_array(inp, m, n):
    img_arr = np.zeros((28*m, 28*n, 3)).astype(np.uint8)

    for i in range(0, m):
        for j in range(0, n):
            img_arr[28*i:28*(i+1), 28*j:28*(j+1)] = inp[i*n+j].reshape((28, 28, 1))*255

    return img_arr

m = 10
n = 10

img_arr = get_img_array(inp_tr, m, n)
img = Image.fromarray(img_arr)
img.save("mnist_numbers_orig.png", "PNG")
# img.show()
# input()

K = 1000

features = 50
D = np.random.random((inp_tr.shape[1], features))
D = D / np.max(np.vstack((np.ones((1, D.shape[1])), np.sqrt(np.sum(D**2, axis=0)).reshape((1, D.shape[1])))), axis=0)
C = np.random.random((features, K))
B = inp_tr[:K, :].T

print("D.shape = {}".format(D.shape))
print("C.shape = {}".format(C.shape))
print("B.shape = {}".format(B.shape))

lambd = 0.01
f = lambda D, C, B, lambd: np.sum((D.dot(C) - B)**2) + lambd*np.sum(np.vectorize(lambda x: np.abs(x))(C[:, 1:]))
error = f(D, C, B, lambd)
errors = [error]
print("error = {}".format(error))

C_prev = C
D_prev = D
for k in range(1, 100):
    beta = (k-1) / (k+2)
    C_tild = C + beta*(C - C_prev)
    D_tild = D + beta*(D - D_prev)
    C_1 = C_tild - 0.2*(2*D.T.dot(D.dot(C_tild) - B) + np.vectorize(lambda x: -1 if x < 0. else 1)(np.hstack((np.zeros((C.shape[0], 1)), C[:, 1:])))) / np.sqrt(np.sum((D.T.dot(D))**2))
    D_1 = D_tild - 0.2*(2*(D.dot(C)-B).dot(C.T)) / np.sqrt(np.sum((C.dot(C.T))**2))

    C_prev = C
    D_prev = D
    C = C_1
    D = D_1
    D = D / np.max(np.vstack((np.ones((1, D.shape[1])), np.sqrt(np.sum(D**2, axis=0)).reshape((1, D.shape[1])))), axis=0)

    error = f(D, C, B, lambd)
    errors.append(error)
    print("k: {}, error = {}".format(k, error))

img_arr = get_img_array((lambda x: (lambda x: x/np.max(x, axis=0))(x-np.min(x, axis=0)))((D.dot(C)).T), m, n)
img = Image.fromarray(img_arr)
img.save("mnist_numbers_features.png", "PNG")
# img.show()
# input()
