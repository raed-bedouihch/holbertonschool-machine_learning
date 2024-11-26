#!/usr/bin/env python3
"""write a function that updates the weights and biases
of a neural network using gradient descent
with L2 regularization"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Returns: the updated weights and
    biases of a neural network"""

    A_prev = cache['A0']
    m = Y.shape[1]
    for lay in range(L, 0, -1):
        A = cache['A' + str(lay)]
        if lay == L:
            der = A - Y
        else:
            der = d_A * (1 - A ** 2)
        if lay > 1:
            d_A = np.dot(weights['W' + str(lay)].T, der)
        A_prev = cache['A' + str(lay - 1)]
        reg_term = (lambtha / m)
        d_W = np.dot(der, A_prev.T) + reg_term * weights['W' + str(lay)]
        d_b = np.sum(der, axis=1, keepdims=True) / m

        weights['W' + str(lay)] -= alpha * d_W
        weights['b' + str(lay)] -= alpha * d_b
    return weights
