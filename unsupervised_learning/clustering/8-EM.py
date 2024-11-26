#!/usr/bin/env python3
"""a function that performs the expectation maximization for a GMM"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """Returns: pi, m, S, g, l, or None, None, None, None, None on failure"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None
    n, d = X.shape
    pi, m, S = initialize(X, k)
    for i in range(iterations):
        g, prev_l = expectation(X, pi, m, S)
        if verbose and i % 10 == 0:
            print(f"Log Likelihood after {i} iterations: {prev_l:.5f}")
        pi, m, S = maximization(X, g)
        g, lo = expectation(X, pi, m, S)
        if np.abs(lo - prev_l) <= tol:
            break
    if verbose:
        print(f"Log Likelihood after {i + 1} iterations: {lo:.5f}")
    return pi, m, S, g, lo
