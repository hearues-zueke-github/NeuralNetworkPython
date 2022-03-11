#! /usr/bin/python2.7

import sys
import os
import gzip
import timeit

import cPickle as pkl
import numpy as np
import Utils as utils

from copy import deepcopy
from random import randint
from DigitRecognition import main_digits_learning_testing_iterations
from TrainedNetwork import TrainedNetwork
from PIL import Image

def test_image_creating():
    img = Image.new("RGB", (256, 256))
    pix = img.load()

    for _ in xrange(0, 1000):
        pix[randint(0, 255), randint(0, 255)] = (randint(0, 255), randint(0, 255), randint(0, 255))

    img.show()
# def test_image_creating
