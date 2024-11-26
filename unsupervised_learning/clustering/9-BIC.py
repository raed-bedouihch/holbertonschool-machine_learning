#!/usr/bin/env python3
"""a function that finds the best number of clusters
for a GMM using the Bayesian Information Criterion"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Returns: best_k, best_result, l, b,
    or None, None, None, None on failure"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None
    if not isinstance(kmax, int) or kmax < kmin:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None
    n, d = X.shape
    if kmax is None:
        kmax = n
    log_li = np.zeros(kmax - kmin + 1)
    bic_val = np.zeros(kmax - kmin + 1)
    best_k = best_result = None
    for k in range(kmin, kmax + 1):
        pi, m, S, g, lo = expectation_maximization(
            X, k, iterations, tol, verbose)
        if lo is None:
            continue
        log_li[k - kmin] = lo
        p = k * (d + d * (d + 1) / 2) + k - 1
        bic_val[k - kmin] = p * np.log(n) - 2 * lo
        if best_k is None or bic_val[k - kmin] < bic_val[best_k - kmin]:
            best_k = k
            best_result = (pi, m, S)
    return best_k, best_result, log_li, bic_val
