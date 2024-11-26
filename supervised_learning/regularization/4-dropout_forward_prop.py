#!/usr/bin/env python3
"""write a function that conducts forward propagation using Dropout"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Returns:  a dictionary containing the outputs
    of each layer and the dropout mask used on each layer """
    cache = {'A0': X}
    for lay in range(1, L + 1):
        W = weights['W' + str(lay)]
        b = weights['b' + str(lay)]
        Z = np.dot(W, cache['A' + str(lay - 1)]) + b
        if lay == L:
            exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        else:
            A = np.tanh(Z)
            dropout_mask = (
                np.random.rand(
                    A.shape[0],
                    A.shape[1]) < keep_prob).astype(int)
            A *= dropout_mask
            A /= keep_prob
            cache['D' + str(lay)] = dropout_mask
        cache['A' + str(lay)] = A
    return cache
