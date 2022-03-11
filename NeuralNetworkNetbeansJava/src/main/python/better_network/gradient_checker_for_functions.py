#! /usr/bin/python2.7

import numpy as np

T = np.array([[0., 0., 1., 1.]])
X = np.array([[0.3, 0.6, 0.1, 0.8]])
Y = np.exp(X) / np.sum(np.exp(X)).reshape((-1, 1))

calc_cecf_one = lambda x, t: np.sum(np.vectorize(lambda xi, ti: np.log(xi) if ti == 1. else np.log(1-xi))(x, t))

def numerical_derivation(X, T):
    derivs = np.zeros_like(X)
    epsilon = 0.0001
    for deriv, x, t in zip(derivs, X, T):
        for i in range(0, len(deriv)):
            x += epsilon
            f1 = calc_cecf_one()

def simple_vector_derivation():
    n = 5
    A = np.random.random((n , n))*5
    x0 = np.random.random((n, ))*2
    b = np.dot(A, x0)
    x = np.random.random((n, ))*2

    f = lambda A, x: np.dot(A, x)+x**2
    calc_mse = lambda A, x, b: np.sum((f(A, x)-b)**2)/2.
    deriv_x = lambda A, x, b: (A.T+np.diag(x)*2).dot(f(A, x)-b)

    def get_numerical_deriv_x(A, x, b):
        x = x.copy()
        x_deriv = x.copy()

        epsilon = 0.0001
        for i in xrange(0, x.shape[0]):
            x[i] += epsilon
            fr = calc_mse(A, x, b)
            x[i] -= epsilon*2
            fl = calc_mse(A, x, b)
            x[i] += epsilon

            x_deriv[i] = (fr-fl) / epsilon / 2.

        return x_deriv

    print("n: {}".format(n))
    print("A:\n{}".format(A))
    print("x0:\n{}".format(x0))
    print("b:\n{}".format(b))
    print("x:\n{}".format(x))

    error_0 = calc_mse(A, x0, b)
    error = calc_mse(A, x, b)

    print("error_0: {}".format(error_0))
    print("error: {}".format(error))

    x_dev = deriv_x(A, x, b)
    print("x_dev:\n{}".format(x_dev))

    x_dev_num = get_numerical_deriv_x(A, x, b)
    print("x_dev_num:\n{}".format(x_dev_num))

def simple_matrix_derivation():
    n = 5
    A = np.random.random((n , n))*5
    X0 = np.random.random((n, n))*2
    B = np.dot(A, X0)
    X = np.random.random((n, n))*2

    # f = lambda A, X: A.dot(X)
    f = lambda A, X: A.dot(X)+X**2
    calc_mse = lambda A, X, B: np.sum((f(A, X)-B)**2)/2.
    # deriv_x = lambda A, X, B: (A.T+X.T*2).dot(A.dot(X)-B)
    # deriv_x = lambda A, X, B: (A.T+X**2).dot(A.dot(X)-B)
    def deriv_x(A, X, B):
        X_new = np.zeros(X.shape)

        for i, (x, b) in enumerate(zip(X.T, B.T)):
            X_new[:, i] = (A.T+np.diag(x)*2).dot(f(A, x)-b)

        return X_new

    def get_numerical_deriv_x(A, X, B):
        X = X.copy()
        X_deriv = X.copy()

        epsilon = 0.0001
        for j in xrange(0, X.shape[0]):
            for i in xrange(0, X.shape[1]):
                X[j, i] += epsilon
                fr = calc_mse(A, X, B)
                X[j, i] -= epsilon*2
                fl = calc_mse(A, X, B)
                X[j, i] += epsilon

                X_deriv[j, i] = (fr-fl) / epsilon / 2.

        return X_deriv

    print("n: {}".format(n))
    print("A:\n{}".format(A))
    print("X0:\n{}".format(X0))
    print("B:\n{}".format(B))
    print("X:\n{}".format(X))

    error_0 = calc_mse(A, X0, B)
    error = calc_mse(A, X, B)

    print("error_0: {}".format(error_0))
    print("error: {}".format(error))

    X_dev = deriv_x(A, X, B)
    print("X_dev:\n{}".format(X_dev))

    X_dev_num = get_numerical_deriv_x(A, X, B)
    print("X_dev_num:\n{}".format(X_dev_num))

def simple_matrix_square_derivation():
    f = lambda A, B, X: np.dot(X, np.dot(A, X))+np.dot(B, X)
    calc_mse = lambda A, B, X, C: np.sum((f(A, B, X)-C)**2)/2.
    
    n = 5
    mult_factor = 5.
    A = np.random.random((n , n))*mult_factor-mult_factor/2.
    B = np.random.random((n , n))*mult_factor-mult_factor/2.
    X0 = np.random.random((n, n))*2-1.
    C = f(A, B, X0)
    X = np.random.random((n, n))*2

    def deriv_x(A, B, X, C):
        X_new = np.zeros(X.shape)

        for i, (x, b) in enumerate(zip(X.T, B.T)):
            X_new[:, i] = (A.T+np.diag(x)*2).dot(f(A, B, x)-C)

        return X_new

    def get_numerical_deriv_x(A, B, X, C):
        X = X.copy()
        X_deriv = X.copy()

        epsilon = 0.0001
        for j in xrange(0, X.shape[0]):
            for i in xrange(0, X.shape[1]):
                X[j, i] += epsilon
                fr = calc_mse(A, B, X, C)
                X[j, i] -= epsilon*2
                fl = calc_mse(A, B, X, C)
                X[j, i] += epsilon

                X_deriv[j, i] = (fr-fl) / epsilon / 2.

        return X_deriv

    print("n: {}".format(n))
    print("A:\n{}".format(A))
    print("X0:\n{}".format(X0))
    print("B:\n{}".format(B))
    print("X:\n{}".format(X))

    error_0 = calc_mse(A, B, X0, C)
    error = calc_mse(A, B, X, C)

    print("error_0: {}".format(error_0))
    print("error: {}".format(error))

    X_dev = deriv_x(A, B, X, C)
    print("X_dev:\n{}".format(X_dev))

    X_dev_num = get_numerical_deriv_x(A, B, X, C)
    print("X_dev_num:\n{}".format(X_dev_num))

# simple_vector_derivation()
simple_matrix_derivation()
# simple_matrix_square_derivation()
