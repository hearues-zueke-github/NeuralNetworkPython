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

fig, ax = plt.subplots(figsize=(10, 6))

rects = []
colors = ["b", "g", "r", "k", "c", "m"]
# for i, rate in enumerate(zip(*rates)):
#     rects.append(ax.bar(ind-width_complete/2.+width_one_bar*i, rate, width_one_bar, color=colors[i]))

x = np.arange(0, 5., 0.1)
funcs_amount = 4
plots = [ax.plot(x, np.random.randint(0, 100, (len(x), )), ".")[0] for _ in xrange(0, funcs_amount)]

ax.set_ylim([0, 100.])
ax.set_title("Test example for legend")
ax.set_xlabel("x")
ax.set_ylabel("y")
# ax.set_xticks(np.arange(0, len(nls_str), 1))
# ax.set_xticklabels(nls_str) # [""]+nls_str)
ax.grid(True)

plt.tight_layout()
plt.subplots_adjust(right=0.9) # , left=0.1, top=0.9, bottom=0.1)
ax.legend(plots, tuple(("f {}".format(i+1) for i in xrange(len(plots)))), bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes, fontsize=12, loc="upper left") # , loc='upper right', ncol=1) #, loc=(0.8, 0.5))
# plt.draw()
plt.savefig("plot_example_legend.png", format="png", dpi=400)
