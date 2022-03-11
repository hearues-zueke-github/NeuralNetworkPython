#! /usr/bin/python2.7

from __init__ import *

f, (ax1, ax2) = plt.subplots(2)

x1 = np.arange(100)
y1 = np.random.random((100,)) * 5

x2 = np.arange(400)
y2 = np.random.random((400,)) * 5

# plt.plot(x1, y1, ".")
ax1.plot(x1, y1, ".")
ax1.set_title("Graph 1")
ax2.plot(x2, y2, "+g")
ax2.set_title("Graph 2")

f.suptitle("Graphs")

plt.savefig("figures/test_plots_1.png", type="png", dpi=500)

f, axarr = plt.subplots(2, 2, figsize=(18,12))

x1 = np.arange(100)
y1 = np.random.random((100,)) * 5

x2 = np.arange(400)
y2 = np.random.random((400,)) * 5

x3 = np.arange(400)
y3 = np.random.random((400,)) * 5

x4 = np.arange(400)
y4 = np.random.random((400,)) * 5

ax = axarr[0, 0]
ax.plot(x1, y1, ".")
ax.set_title("Graph 1")

ax = axarr[0, 1]
ax.plot(x2, y2, "+g")
ax.set_title("Graph 2")

ax = axarr[1, 0]
ax.plot(x1, y1, ".", color="#00FFCC")
ax.set_yscale("log", nonposy='clip')
ax.set_title("Graph 3")

ax = axarr[1, 1]
ax.plot(x2, y2, "+g", color="#CC88A0")
ax.set_title("Graph 4")

# f.suptitle("Graphs")

plt.savefig("figures/test_plots_2.png", type="png", dpi=500)
