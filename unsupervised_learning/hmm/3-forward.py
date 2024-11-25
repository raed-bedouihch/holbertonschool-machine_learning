#!/usr/bin/env python3
"""a function that performs the forward algorithm for a hidden markov model"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """P, F, or None, None on failure"""
    try:
        T = Observation.shape[0]
        N = Emission.shape[0]
        F = np.zeros((N, T))
        F[:, 0] = Initial.flatten() * Emission[:, Observation[0]]
        for t in range(1, T):
            for n in range(N):
                F[n, t] = np.sum(F[:, t - 1] * Transition[:, n]
                                 * Emission[n, Observation[t]])
        P = np.sum(F[:, -1])
        return P, F
    except BaseException:
        return None, None
