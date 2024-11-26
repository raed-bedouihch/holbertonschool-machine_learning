#!/usr/bin/env python3
"""a function that calculates the probability density
function of a Gaussian distribution"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """calculates the expectation step in the EM algorithm for a GMM"""
    try:
        n, d = X.shape
        if not np.isclose(np.sum(pi), 1):
            return None, None
        k = len(pi)
        g = np.zeros((k, n))
        for i in range(k):
            g[i] = pi[i] * pdf(X, m[i], S[i])
        g_sum = np.sum(g, axis=0)
        g /= g_sum
        tot_l = np.sum(np.log(g_sum))
        return g, tot_l
    except BaseException:
        return None, None
