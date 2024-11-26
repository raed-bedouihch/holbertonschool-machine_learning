#!/usr/bin/env python3
"""a function that performs PCA on a dataset"""
import numpy as np


def pca(X, var=0.95):
    """Returns: the weights matrix, W,
    that maintains var fraction of X‘s original variance"""

    U, s, Vt = np.linalg.svd(X)
    cum_var_ratio = np.cumsum(s) / np.sum(s)
    r = 0
    for i, v in enumerate(cum_var_ratio):
        if v >= var:
            r = i + 1
            break
    W = np.transpose(Vt[:r])
    return W
