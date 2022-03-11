#! /usr/bin/python2.7

import numpy as np

import matplotlib.pyplot as plt

from PIL import Image

calc_mse = lambda x, A, t: np.sum((A.dot(x)-t)**2)

m, n = 10, 100

x = np.random.random(n)
A = np.random.random((m, n))
t = A.dot(x) # np.random.random(m)
x = np.random.random(n)
mses = [calc_mse(x, A, t)]

# print("x =\n{}".format(x))
# print("A =\n{}".format(A))
# print("t =\n{}".format(t))

save_weights_amount = 1000
weights_history = np.zeros((save_weights_amount, x.shape[0]))

eta = 1. / n / 10.
iterations = 10000
for i in xrange(0, iterations):
    if i >= save_weights_amount:
        break
    dx = A.T.dot(A.dot(x)-t)
    weights_history[i] = dx
    # print("dx =\n{}".format(dx))
    x -= eta*dx
    mse = calc_mse(x, A, t)
    mses.append(mse)
    if mses[-2] - mse < 0.0001:
        break
    print("i: {}, mse: {}".format(i, mse))

I = np.zeros((save_weights_amount, x.shape[0], 3)) # for RGB
def get_condition_values_only(X, pos):
    Xn = np.zeros_like(X)
    Xn[pos] = X[pos]
    return Xn
X_p = get_condition_values_only(weights_history, np.where(weights_history>=0.))
X_m = -get_condition_values_only(weights_history, np.where(weights_history<0.))

X_p = X_p/np.max(X_p)*255
X_m = X_m/np.max(X_m)*255

I[:, :, 0] = X_p
I[:, :, 2] = X_m
I = I.astype(np.uint8)
img = Image.fromarray(I)
img.save("min_max_pixels.png", "PNG")

plt.figure()
plt.title("MSEs")
plt.plot(np.arange(0, len(mses)), mses, "b-")
plt.show()
