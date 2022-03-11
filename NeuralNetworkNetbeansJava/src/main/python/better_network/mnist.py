#! /usr/bin/python2.7

import sys
import cPickle
import gzip
import numpy as np

# importing a file, which is parent
sys.path.append("..")

np.set_printoptions(threshold=np.nan)

# Load the dataset
with gzip.open('mnist.pkl.gz', 'rb') as f:
    (X_train, V_train), (X_valid, V_valid), (X_test, V_test) = cPickle.load(f)

V_train = V_train.astype(np.int)
V_valid = V_valid.astype(np.int)
V_test = V_test.astype(np.int)

print("type(X_train) = "+str(type(X_train)))
print("type(X_valid) = "+str(type(X_valid)))
print("type(X_test)  = "+str(type(X_test)))

print("type(V_train) = "+str(type(V_train)))
print("type(V_valid) = "+str(type(V_valid)))
print("type(V_test)  = "+str(type(V_test)))

print("X_train.shape = "+str(X_train.shape))
print("X_valid.shape = "+str(X_valid.shape))
print("X_test.shape  = "+str(X_test.shape))

print("V_train.shape = "+str(V_train.shape))
print("V_valid.shape = "+str(V_valid.shape))
print("V_test.shape  = "+str(V_test.shape))

def half_grayscale_image(array, y, x):
    return np.mean(array.reshape((y//2, 2, x)).transpose(0, 2, 1).reshape((y//2, x//2, 4)), axis=-1)
def half_grayscale_images(array, y, x):
    l = array.shape[0]
    return np.mean(array.reshape((l, y//2, 2, x)).transpose(0, 1, 3, 2).reshape((l, y//2, x//2, 4)), axis=-1).reshape((l, -1))

print("convert to onehot array")
T_train = np.zeros((V_train.shape[0], 10))
T_train[np.arange(0, V_train.shape[0]), V_train] = 1.
T_valid = np.zeros((V_valid.shape[0], 10))
T_valid[np.arange(0, V_valid.shape[0]), V_valid] = 1.
T_test = np.zeros((V_test.shape[0], 10))
T_test[np.arange(0, V_test.shape[0]), V_test] = 1.

print("convert to 14x14 images")
X_train_14 = half_grayscale_images(X_train, 28, 28)
X_valid_14 = half_grayscale_images(X_valid, 28, 28)
X_test_14 = half_grayscale_images(X_test, 28, 28)

print("convert to 7x7 images")
X_train_7 = half_grayscale_images(X_train_14, 14, 14)
X_valid_7 = half_grayscale_images(X_valid_14, 14, 14)
X_test_7 = half_grayscale_images(X_test_14, 14, 14)

print("Saving mnist 28x28 as mnist_28x28.npz.gz")
with open("mnist_28x28.npz.gz", "wb") as f:
    np.savez_compressed(f, X_train=X_train, T_train=T_train, V_train=V_train,
                           X_valid=X_valid, T_valid=T_valid, V_valid=V_valid,
                           X_test=X_test, T_test=T_test, V_test=V_test)

print("Saving mnist 14x14 as mnist_14x14.npz.gz")
with open("mnist_14x14.npz.gz", "wb") as f:
    np.savez_compressed(f, X_train=X_train_14, T_train=T_train, V_train=V_train,
                           X_valid=X_valid_14, T_valid=T_valid, V_valid=V_valid,
                           X_test=X_test_14, T_test=T_test, V_test=V_test)

print("Saving mnist 7x7 as mnist_7x7.npz.gz")
with open("mnist_7x7.npz.gz", "wb") as f:
    np.savez_compressed(f, X_train=X_train_7, T_train=T_train, V_train=V_train,
                           X_valid=X_valid_7, T_valid=T_valid, V_valid=V_valid,
                           X_test=X_test_7, T_test=T_test, V_test=V_test)
