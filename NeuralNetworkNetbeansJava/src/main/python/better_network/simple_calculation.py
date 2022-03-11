#! /usr/bin/python3

import numpy as np

def print_var(x):
    print(x+": "+str(globals()[x]))

def get_new_theta(X, y, theta, etha):
    grad = np.zeros((theta.size, 1))
    # print("grad = "+str(grad))
    for j in range(0, theta.size):
        for i in range(0, X.shape[0]):
            grad[j] += (np.dot(X[i], theta) - y[i]) * X[i][j]

    # print("grad = "+str(grad))
    return theta - (2*etha*(1/float(X.shape[0])))*grad

m = 2
n = 5

X = np.hstack((np.ones((m, 1)), np.random.randint(-10, 10, (m, n-1))))
print_var("X")

theta = np.random.randint(-10, 10, (n, 1))
# print_var("theta")

h = np.dot(X, theta)
print_var("h")

y = np.random.randint(-10, 10, (m, 1))
print_var("y")

diff = h - y
print_var("diff")

print("Now the iterations:")

etha = 0.005
print(X.shape[0])

new_theta = theta
for i in range(0, 1000):
    # print("Iteration nr. "+str(i))
    new_theta = get_new_theta(X, y, theta, etha)

new_h = np.dot(X, new_theta)

print("final values")    
# print_var("new_theta")
print_var("new_h")


