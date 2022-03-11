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

import matplotlib.pyplot as plt
import matplotlib.ticker as tkr

from copy import deepcopy
from PIL import Image

x = np.arange(0, 5, 0.1)

fig, ax = plt.subplots(3, 1, figsize=(10, 15))

plt.suptitle("My sup title", fontsize=16)

plots = []
ax[0].set_title("Title 1")
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
plots.append(ax[0].plot(x, x*7+3, "b.")[0])
plots.append(ax[0].plot(x, x*2+3, "g.")[0])
plots.append(ax[0].plot(x, -x*2+5, "r.")[0])
ax[0].legend(plots, ("f1", "f2", "f3"), bbox_to_anchor=(1, 1), loc="upper left", fontsize=12, title="Functions")

plots = []
ax[1].set_title("Title 2")
ax[1].set_xlabel("x")
ax[1].set_ylabel("y")
plots.append(ax[1].plot(x, x*2+3, "b.")[0])
plots.append(ax[1].plot(x, x*3+3, "g.")[0])
plots.append(ax[1].plot(x, -x*4+5, "r.")[0])
ax[1].legend(plots, ("f1", "f2", "f3"), bbox_to_anchor=(1, 1), loc="upper left", fontsize=12)

ax[2].set_title("Title 3")
ax[2].set_xlabel("x")
ax[2].set_ylabel("y")
plots.append(ax[2].plot(x, x*9+3, "b.")[0])
plots.append(ax[2].plot(x, -x*6+3, "g.")[0])
plots.append(ax[2].plot(x, -x*2+5, "r.")[0])
ax[2].legend(plots, ("f1", "f2", "f3"), bbox_to_anchor=(1, 1), loc="upper left", fontsize=12)

plt.tight_layout()
# ax[0].subplots_adjust(top=0.7)
plt.subplots_adjust(top=0.9, right=0.88, hspace=0.25) # , left=0.1, top=0.9, bottom=0.1)
# plt.show()
plt.savefig("example_test_plot.png", type="png", dpi=300)
