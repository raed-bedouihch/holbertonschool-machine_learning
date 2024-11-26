#!/usr/bin/env python3
"""write a function that updates the weights
of a neural network with Dropout regularization
using gradient descent"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """All layers use thetanh activation function
    except the last, which uses the softmax activation function"""
    classes, m = Y.shape
    A_prev = cache['A' + str(L - 1)]
    curr_A = cache['A' + str(L)]
    dZ = curr_A - Y
    for lay in range(L, 0, -1):
        A_prev = cache['A' + str(lay - 1)]
        W = weights['W' + str(lay)]
        b = weights['b' + str(lay)]
        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        if lay >= 2:
            d_A_prev = np.dot(W.T, dZ)
            dropout_mask = cache['D' + str(lay - 1)]
            d_A_prev *= dropout_mask
            d_A_prev /= keep_prob
            dZ = d_A_prev * (1 - A_prev ** 2)
        weights['W' + str(lay)] = W - alpha * dW
        weights['b' + str(lay)] = b - alpha * db
