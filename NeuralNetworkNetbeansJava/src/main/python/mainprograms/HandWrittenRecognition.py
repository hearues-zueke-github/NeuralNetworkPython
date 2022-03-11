#! /usr/bin/python2.7

from __init__ import *

# TODO call an other function rounded and this create 14x14
def create_raw_rounded_14x14_train_valid_test():
    """
        This wil create raw data sets of train, valid and test and save it as lists
    """
    # Load the training pictures from MNIST and give it to the neural network
    with gzip.open("mnist.pkl.gz", 'rb') as f:
        print("Loading mnist package in train_set, valid_set and test_set")
        train_set, valid_set, test_set = pkl.load(f)

    print("Loaded, convert to lists!")

    temp = train_set
    train_set = [[list(t) for t in temp[0]], list(temp[1])]
    temp = valid_set
    valid_set = [[list(t) for t in temp[0]], list(temp[1])]
    temp = test_set
    test_set = [[list(t) for t in temp[0]], list(temp[1])]

    print("Modify lists!")

    train_set[0] = [[0. if temp2 < 0.5 else 1. for temp2 in temp1] for temp1 in train_set[0]]
    valid_set[0] = [[0. if temp2 < 0.5 else 1. for temp2 in temp1] for temp1 in valid_set[0]]
    test_set[0] = [[0. if temp2 < 0.5 else 1. for temp2 in temp1] for temp1 in test_set[0]]

    # Transform all 28x28 pictures to 14x14
    temp_all = []
    for temp, i in zip(train_set[0], xrange(0, len(train_set[0]))):
        temp2 = [temp[28*i:28*(i+1)] for i in xrange(0, 28)]

        temp3 = [[0. for _ in xrange(0, 14)] for _ in xrange(0, 14)]

        for y in xrange(0, 14):
            for x in xrange(0, 14):
                for yi in xrange(0, 2):
                    for xi in xrange(0, 2):
                        temp3[y][x] += temp2[2*y+yi][2*x+xi]
                temp3[y][x] /= 4.

        temp4 = []
        for t in temp3:
            temp4 += t

        temp_all.append(temp4)
    train_set[0] = temp_all

    temp_all = []
    for temp, i in zip(valid_set[0], xrange(0, len(valid_set[0]))):
        temp2 = [temp[28*i:28*(i+1)] for i in xrange(0, 28)]

        temp3 = [[0. for _ in xrange(0, 14)] for _ in xrange(0, 14)]

        for y in xrange(0, 14):
            for x in xrange(0, 14):
                for yi in xrange(0, 2):
                    for xi in xrange(0, 2):
                        temp3[y][x] += temp2[y+yi][x+xi]
                temp3[y][x] /= 4.

        temp4 = []
        for t in temp3:
            temp4 += t

        temp_all.append(temp4)
    valid_set[0] = temp_all

    temp_all = []
    for temp, i in zip(test_set[0], xrange(0, len(test_set[0]))):
        temp2 = [temp[28*i:28*(i+1)] for i in xrange(0, 28)]

        temp3 = [[0. for _ in xrange(0, 14)] for _ in xrange(0, 14)]

        for y in xrange(0, 14):
            for x in xrange(0, 14):
                for yi in xrange(0, 2):
                    for xi in xrange(0, 2):
                        temp3[y][x] += temp2[y+yi][x+xi]
                temp3[y][x] /= 4.

        temp4 = []
        for t in temp3:
            temp4 += t

        temp_all.append(temp4)
    test_set[0] = temp_all

    for i in xrange(0, 14):
        print(str(train_set[0][0][14*i:14*(i+1)]))

    with gzip.GzipFile("set_train_raw_rounded_14x14.pkl.gz", "wb") as fout:
        pkl.dump(train_set, fout)
    with gzip.GzipFile("set_valid_raw_rounded_14x14.pkl.gz", "wb") as fout:
        pkl.dump(valid_set, fout)
    with gzip.GzipFile("set_test_raw_rounded_14x14.pkl.gz", "wb") as fout:
        pkl.dump(test_set, fout)
# def create_raw_train_valid_test

def create_autoencoder_inputs_targets_14x14():
    """
    This will create inputs and targets for an autoencoder,
    where input and target are the same
    """
    print("Load files!")
    with gzip.GzipFile("set_train_raw_rounded_14x14.pkl.gz", "rb") as fin:
        train_set = pkl.load(fin)
    with gzip.GzipFile("set_valid_raw_rounded_14x14.pkl.gz", "rb") as fin:
        valid_set = pkl.load(fin)
    with gzip.GzipFile("set_test_raw_rounded_14x14.pkl.gz", "rb") as fin:
        test_set = pkl.load(fin)

    # print("Before func function:")
    # for i in xrange(0, 14):
    #     print(str(train_set[0][0][14*i:14*(i+1)]))

    func = lambda x: 0.5 + (x - 0.5) * 0.8
    print("Change all values with function: f(x) = 0.5 + (x - 0.5) * 0.8")
    train_set[0] = [[func(t) for t in temp] for temp in train_set[0]]
    valid_set[0] = [[func(t) for t in temp] for temp in valid_set[0]]
    test_set[0]  = [[func(t) for t in temp] for temp in test_set[0]]

    # print("After func function:")
    # for i in xrange(0, 14):
    #     print(str(train_set[0][0][14*i:14*(i+1)]))

    # Creating for autoencoder the files
    print("Convert lists to numpy array and inputs targets value")
    temp = [[], [], []]
    for temp1, temp2 in zip(train_set[0], train_set[1]):
        temp[0].append(np.array([temp1]).transpose())
        temp[1].append(np.array([temp1]).transpose())
        temp[2].append(temp2)
    train_set = temp
    print("Finished 14x14 train autoencoder")

    temp = [[], [], []]
    for temp1, temp2 in zip(valid_set[0], valid_set[1]):
        temp[0].append(np.array([temp1]).transpose())
        temp[1].append(np.array([temp1]).transpose())
        temp[2].append(temp2)
    valid_set = temp
    print("Finished 14x14 valid autoencoder")

    temp = [[], [], []]
    for temp1, temp2 in zip(test_set[0], test_set[1]):
        temp[0].append(np.array([temp1]).transpose())
        temp[1].append(np.array([temp1]).transpose())
        temp[2].append(temp2)
    test_set = temp
    print("Finished 14x14 test autoencoder")

    with gzip.GzipFile("inputs_targets_14x14_autoencoder_train.pkl.gz", "wb") as fout:
        pkl.dump(train_set, fout)
    print("Saved train set")
    with gzip.GzipFile("inputs_targets_14x14_autoencoder_valid.pkl.gz", "wb") as fout:
        pkl.dump(valid_set, fout)
    print("Saved valid set")
    with gzip.GzipFile("inputs_targets_14x14_autoencoder_test.pkl.gz", "wb") as fout:
        pkl.dump(test_set, fout)
    print("Saved test set")
# def create_autoencoder_inputs_targets_14x14
