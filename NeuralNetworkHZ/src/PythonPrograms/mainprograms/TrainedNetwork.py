#! /usr/bin/python2.7

from NeuralNetworkDecimalMultiprocess import NeuralNetwork

from decimal import Decimal as dec
from copy import deepcopy

import numpy as np

import pickle
import gzip
import os

class TrainedNetwork(Exception):
    error_no_path_set = "Cannot load network, no path set!"
    error_no_valid_path = "Cannot laod network, no valid path!"
    error_no_access_to_write = "Not able to save network, no access for given path!"
    error_not_network_type = "Given neural network is not instance of 'NeuralNetwork'!"
    error_not_a_string = "Given path is not a String!"

    def __init__(self):
        self.is_path_set = False
        self.file_path = ""

        self.is_network_loaded = False
        self.nn = None
    # def __init__

    def get_file_path(self):
        return self.file_path
    # def

    def set_file_path(self, file_path):
        if not isinstance(file_path, str):
            print(self.error_not_a_string)
            return -1
        # if

        self.file_path = file_path
        self.is_path_set = True

        return 0
    # def sef_file_path

    def load_network(self):
        if not self.is_path_set:
            print(self.error_no_path_set)
            return -1
        # if

        if not os.path.isfile(self.file_path):
            print(self.error_no_valid_path)
            return -1
        # if

        with gzip.open(self.file_path, 'rb') as fi:
            self.nn = pickle.load(fi)
        # with
        # self.nn = pickle.load(self.file_path)
        
        return 0
    # def load_network

    def save_network(self):
        if not self.is_path_set:
            print(self.error_no_path_set)
            return -1
        # if

        # maybe not needed
        # if not os.access(self.file_path, os.W_OK):
        #     print(self.error_no_access_to_write)
        #     return -1
        # # if

        with gzip.GzipFile(self.file_path, 'wb') as fo:
        	# -1 == like pickle.HIGHEST_PROTOCOL
            pickle.dump(self.nn, fo, -1)
        # with
        # pickle.dump(self.nn, self.file_path)

        return 0
    # def save_network

    def get_network(self):
        return self.nn
    # def get_network

    def set_network(self, nn):
        if not isinstance(nn, NeuralNetwork):
            print(self.error_not_network_type)
            return -1
        # if

        self.nn = deepcopy(nn)

        return 0
    # def set_network
# class TrainedNetwork
