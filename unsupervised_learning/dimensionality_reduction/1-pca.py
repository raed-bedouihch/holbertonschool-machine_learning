#!/usr/bin/env python3
"""a function that performs PCA on a dataset"""
import numpy as np


def pca(X, ndim):
    """T, a numpy.ndarray of shape (n, ndim) containing
    the transformed version of X"""
    X_mean = X - np.mean(X, axis=0)
    U, s, Vt = np.linalg.svd(X_mean)
    T = np.dot(X_mean, Vt[:ndim].T)
    return T
