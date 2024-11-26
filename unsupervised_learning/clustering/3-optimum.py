#!/usr/bin/env python3
"""a function that tests for the optimum number of clusters by variance"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """results, d_vars, or None, None on failure"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None
    if not isinstance(kmax, int) or kmax < kmin:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    if kmax <= kmin:
        return None, None
    if kmax is None:
        max_clu = X.shape[0]
    else:
        max_clu = kmax
    results = []
    d_vars = []
    C, clss = kmeans(X, kmin, iterations)
    var = variance(X, C)
    results.append((C, clss))
    for k in range(kmin + 1, max_clu + 1):
        C, clss = kmeans(X, k, iterations)
        curr_var = variance(X, C)
        results.append((C, clss))
        d_vars.append(var - curr_var)
    d_vars.insert(0, 0.0)
    return results, d_vars
