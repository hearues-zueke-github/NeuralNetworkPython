#! /usr/bin/python2.7

import cPickle
import gzip
import os
import sys

import numpy as np

printf = sys.stdout.write

# importing a file, which is parent
sys.path.append("..")

np.set_printoptions(threshold=np.nan)

# Load the dataset
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
print("loaded cPickle file!")
f.close()

train_set = list(train_set)
valid_set = list(valid_set)
test_set = list(test_set)

train_set[1] = train_set[1].astype(np.int8)
valid_set[1] = valid_set[1].astype(np.int8)
test_set[1] = test_set[1].astype(np.int8)

print("type(train_set) = "+str(type(train_set)))
print("type(valid_set) = "+str(type(valid_set)))
print("type(test_set) = "+str(type(test_set)))

print("type(train_set[0]) = "+str(type(train_set[0])))
print("type(valid_set[0]) = "+str(type(valid_set[0])))
print("type(test_set[0]) = "+str(type(test_set[0])))

print("type(train_set[0][0]) = "+str(type(train_set[0][0])))
print("type(valid_set[0][0]) = "+str(type(valid_set[0][0])))
print("type(test_set[0][0]) = "+str(type(test_set[0][0])))

print("type(train_set[0][0][0]) = "+str(type(train_set[0][0][0])))
print("type(valid_set[0][0][0]) = "+str(type(valid_set[0][0][0])))
print("type(test_set[0][0][0]) = "+str(type(test_set[0][0][0])))

print("type(train_set[1]) = "+str(type(train_set[1])))
print("type(valid_set[1]) = "+str(type(valid_set[1])))
print("type(test_set[1]) = "+str(type(test_set[1])))

print("type(train_set[1][0]) = "+str(type(train_set[1][0])))
print("type(valid_set[1][0]) = "+str(type(valid_set[1][0])))
print("type(test_set[1][0]) = "+str(type(test_set[1][0])))

print("train_set = "+str(len(train_set)))
print("valid_set = "+str(len(valid_set)))
print("test_set = "+str(len(test_set)))

print("train_set[0] = "+str(len(train_set[0])))
print("valid_set[0] = "+str(len(valid_set[0])))
print("test_set[0] = "+str(len(test_set[0])))

print("train_set[1] = "+str(len(train_set[1])))
print("valid_set[1] = "+str(len(valid_set[1])))
print("test_set[1] = "+str(len(test_set[1])))

print("train_set[0][0] = "+str(len(train_set[0][0])))
print("valid_set[0][0] = "+str(len(valid_set[0][0])))
print("test_set[0][0] = "+str(len(test_set[0][0])))

print("Create matrices for the mnist data! (for targets)")

# inp_tr = np.zeros((50000, 28*28))
# inp_vld = np.zeros((10000, 28*28))
# inp_tst = np.zeros((10000, 28*28))

targ_tr = np.zeros((50000, 10))
targ_vld = np.zeros((10000, 10))
targ_tst = np.zeros((10000, 10))

for i, num in enumerate(train_set[1]):
    targ_tr[i][num] = 1
for i, num in enumerate(valid_set[1]):
    targ_vld[i][num] = 1
for i, num in enumerate(test_set[1]):
    targ_tst[i][num] = 1

train_set[1] = targ_tr
valid_set[1] = targ_vld
test_set[1] = targ_tst

directory = "../mainprograms/original_sets/"
if not os.path.exists(directory):
    os.makedirs(directory)

print("Now saving the sets seperated as orig_<>.pkl.gz")

with gzip.GzipFile(directory+"train_set.pkl.gz", "wb") as fout:
    cPickle.dump(train_set, fout)
with gzip.GzipFile(directory+"valid_set.pkl.gz", "wb") as fout:
    cPickle.dump(valid_set, fout)
with gzip.GzipFile(directory+"test_set.pkl.gz", "wb") as fout:
    cPickle.dump(test_set, fout)

# with gzip.GzipFile(directory+"inp_targ_train.pkl.gz", "wb") as fout:
#     cPickle.dump([inp_tr, targ_tr], fout)
# with gzip.GzipFile(directory+"inp_targ_valid.pkl.gz", "wb") as fout:
#     cPickle.dump([inp_vld, targ_vld], fout)
# with gzip.GzipFile(directory+"inp_targ_test.pkl.gz", "wb") as fout:
#     cPickle.dump([inp_tst, targ_tst], fout)
