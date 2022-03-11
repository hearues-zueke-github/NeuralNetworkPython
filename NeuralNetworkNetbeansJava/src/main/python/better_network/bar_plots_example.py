#! /usr/bin/python2.7

import csv
import gzip
import os
import select
import sys
import time

import numpy as np
import multiprocessing as mp
import pickle as pkl

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr

from copy import deepcopy
from PIL import Image

bits = 4
archs_amount = 5
architectures = [[bits*2, bits+1+i, bits+1] for i in xrange(0, archs_amount)]

train_error = [80.-i*15 for i in xrange(0, archs_amount)]
test_error = [90.-i*13 for i in xrange(0, archs_amount)]

ind = np.arange(archs_amount)
width = 0.3

fig, ax = plt.subplots()

rects1 = ax.bar(ind-width, train_error, width, color="b")
rects2 = ax.bar(ind, test_error, width, color="g")

ax.set_ylim([0, 100.])
ax.set_xlabel("architectures")
ax.set_ylabel("misclass rate [%]")
ax.set_title("Misclass rate for binary adder, {} bits".format(bits))
ax.set_xticklabels([""]+list(map(lambda x: "_".join(map(str, x)), architectures)))

ax.legend((rects1[0], rects2[0]), ("train error", "test error"))

plt.savefig("bar_plots_example.png", format="png", dpi=400)
