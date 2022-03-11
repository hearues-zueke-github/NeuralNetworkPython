#! /usr/bin/python2.7

import sys
import cPickle
import gzip
import numpy as np

printf = sys.stdout.write

# importing a file, which is parent
sys.path.append("..")
# from NeuralNetworkBakk import NeuralNetwork

np.set_printoptions(threshold=np.nan)

# Load the dataset
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)

f.close()

print("type(train_set) = "+str(type(train_set)))
print("type(valid_set) = "+str(type(valid_set)))
print("type(test_set) = "+str(type(test_set)))

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

# t1 = test_set[0][0].tolist()
# for y in xrange(0, 28):
#   for x in xrange(0, 28):
#     if t1[x + y * 28] >= 0.5:
#       sys.stdout.write("1")
#     else:
#       sys.stdout.write("0")
#   print("")

print("Doing a deep copy")
# trainset = np.array(train_set[0], copy=True)
trainset = [["1" if t >= 0.5 else "0" for t in ts.tolist()] for ts in train_set[0]]
numberset = [x for x in train_set[1]]
print("changing values")

for i in xrange(0, 50):
  printf("#"+str(numberset[i])+"\n")
  for j in xrange(0, 28):
    printf(''.join(trainset[i][j * 28:(j+1)*28])+str("\n"))
  printf("\n")

# for t1 in trainset[0]:
#   for t2 in t1:
#     for t3 in np.nditer(t2.T):
#       t3[...] = 1 if t3 >= 0.5 else 0
#   t2 = t2copy

# print("printing the vlaues")
      
# print(str(trainset[0]))

# nn = NeuralNetwork()
