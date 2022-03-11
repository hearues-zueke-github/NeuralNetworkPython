#! /usr/bin/python2.7

# System wide imports
import collections
import gzip
import math
import matplotlib
import os
import shutil
import sys
import time
import timeit
import PIL

matplotlib.use("Agg")

import pylab
import shutil
import time

import cPickle as pkl
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np

from PIL import Image, ImageDraw
from copy import deepcopy
from random import randint
from multiprocessing import Queue, Manager, Process, Lock

# My own imports
import BinaryAdder as binadd
import DigitRecognition as digitrecognition
import Utils as utils

from DigitRecognition import main_digits_learning_testing_iterations
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from NeuralNetworkDecimalMultiprocess import NeuralNetwork
from TrainedNetwork import TrainedNetwork
