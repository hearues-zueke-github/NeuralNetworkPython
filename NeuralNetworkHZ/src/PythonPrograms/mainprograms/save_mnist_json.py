#! /usr/bin/python2.7

import sys
import os
import cPickle
import gzip
import json
# import numpy as np

printf = sys.stdout.write

# importing a file, which is parent
# sys.path.append("..")
# from NeuralNetworkBakk import NeuralNetwork

# np.set_printoptions(threshold=np.nan)

create_dir = lambda directory: 0 if os.path.exists(directory) else os.makedirs(directory)

# Load the dataset
f = gzip.open('mnist/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)

f.close()

print("type(train_set) = "+str(type(train_set)))
print("len(train_set) = "+str(len(train_set)))
print("type(train_set[0]) = "+str(type(train_set[0])))
print("len(train_set[0]) = "+str(len(train_set[0])))
print("type(train_set[1]) = "+str(type(train_set[1])))
print("len(train_set[1]) = "+str(len(train_set[1])))
print("len(train_set[0][0]) = "+str(len(train_set[0][0])))
print("type(train_set[0][0]) = "+str(type(train_set[0][0])))
print("type(train_set[1][0]) = "+str(type(train_set[1][0])))

array, valid_array, test_array = [], [], []
train_array_value = [int(i) for i in train_set[1]]
valid_array_value = [int(i) for i in valid_set[1]]
test_array_value = [int(i) for i in test_set[1]]

filepath = os.getcwd()
json_dir = filepath+"/../../../jsonfiles/"
mnist_path = json_dir + "mnist_path/"
create_dir(mnist_path)

# train: 50.000 sampels
train_path = mnist_path + "train_path/"
create_dir(train_path)

for i in xrange(0, 50):
  print("processing train nr. "+str(i))
  absolute_path = train_path + "train_"+str(i)+".json.gz"
  with gzip.GzipFile(absolute_path, "w") as fout:
    array = []
    array_value = []
    for i, (a, v) in enumerate(zip(train_set[0][i*(1000):(i+1)*1000], train_set[1][i*(1000):(i+1)*1000])):
      array.append([float(a[y * 28 + x]) for x in xrange(0, 28) for y in xrange(0, 28)])
      array_value.append(float(v))
    fout.write(json.dumps([array, array_value]))

# valid: 10.000 sampels
valid_path = mnist_path + "valid_path/"
create_dir(valid_path)

for i in xrange(0, 10):
  print("processing train nr. "+str(i))
  absolute_path = valid_path + "valid_"+str(i)+".json.gz"
  with gzip.GzipFile(absolute_path, "w") as fout:
    array = []
    array_value = []
    for i, (a, v) in enumerate(zip(valid_set[0][i*(1000):(i+1)*1000], valid_set[1][i*(1000):(i+1)*1000])):
      array.append([float(a[y * 28 + x]) for x in xrange(0, 28) for y in xrange(0, 28)])
      array_value.append(float(v))
    fout.write(json.dumps([array, array_value]))

# test : 10.000 sampels
test_path = mnist_path + "test_path/"
create_dir(test_path)

for i in xrange(0, 10):
  print("processing train nr. "+str(i))
  absolute_path = test_path + "test_"+str(i)+".json.gz"
  with gzip.GzipFile(absolute_path, "w") as fout:
    array = []
    array_value = []
    for i, (a, v) in enumerate(zip(test_set[0][i*(1000):(i+1)*1000], test_set[1][i*(1000):(i+1)*1000])):
      array.append([float(a[y * 28 + x]) for x in xrange(0, 28) for y in xrange(0, 28)])
      array_value.append(float(v))
    fout.write(json.dumps([array, array_value]))

print("Finished with writting JSON file")
