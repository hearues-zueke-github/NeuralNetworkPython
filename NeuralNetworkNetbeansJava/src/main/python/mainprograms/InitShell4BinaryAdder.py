#! /usr/bin/python2.7

from __init__ import *
# import sys
# import os

# from TrainedNetwork import TrainedNetwork
# from BinaryAdder import *


tn = TrainedNetwork()

tn.set_file_path("binary_adder_3_bit.pkl.gz")

tn.load_network()

nn = tn.nn

def main(argv):
    pass
# def main

if __name__ == "__main__":
    main(sys.argv)
# if