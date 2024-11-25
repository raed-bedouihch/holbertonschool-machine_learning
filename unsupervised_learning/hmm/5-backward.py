#!/usr/bin/env python3
"""a function that  performs the backward
algorithm for a hidden markov model"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """Returns:  P, B, or None, None on failure"""
    try:
        T = Observation.shape[0]
        N = Emission.shape[0]
        B = np.zeros((N, T))
        B[:, T - 1] = 1
        for t in range(T - 2, -1, -1):
            for n in range(N):
                B[n, t] = np.sum(B[:, t + 1] * Transition[n, :]
                                 * Emission[:, Observation[t + 1]])
        P = np.sum(Initial.flatten() * Emission[:, Observation[0]] * B[:, 0])
        return P, B
    except BaseException:
        return None, None
