#!/usr/bin/env python3
""" a function that determines the probability of a markov chain
being in a particular state after a specified number of iterations"""
import numpy as np


def markov_chain(P, s, t=1):
    """Returns: a numpy.ndarray of shape (1, n) representing the
    probability of being in a specific state after t iterations
    or None on failure"""
    if not isinstance(P, np.ndarray) or not isinstance(s, np.ndarray):
        return None
    if P.shape[0] != P.shape[1]:
        return None
    if s.shape[1] != P.shape[0]:
        return None
    for i in range(t):
        s = np.dot(s, P)
    return s
