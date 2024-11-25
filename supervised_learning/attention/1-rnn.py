#!/usr/bin/env python3
"""a function that performs forward propagation for a simple RNN"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """Returns: H, Y"""
    t, m, i = X.shape
    h = h_0.shape[1]
    H = np.zeros((t + 1, m, h))
    H[0] = h_0
    Y = []
    h_prev = h_0
    for i in range(t):
        h_next, y = rnn_cell.forward(h_prev, X[i])
        H[i + 1] = h_next
        Y.append(y)
        h_prev = h_next
    return H, np.array(Y)
