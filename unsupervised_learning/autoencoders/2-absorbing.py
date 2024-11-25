#!/usr/bin/env python3
"""a function that determines if a markov chain is absorbing"""
import numpy as np


def absorbing(P):
    """Returns: True if it is absorbing, or False on failure"""
    if not isinstance(
            P,
            np.ndarray) or P.ndim != 2 or P.shape[0] != P.shape[1]:
        return False
    n = P.shape[0]
    absorbing_states = np.all(P == np.eye(n), axis=1)
    if not np.any(absorbing_states):
        return False
    non_absorbing = P[~absorbing_states][:, ~absorbing_states]
    identity_mat = np.eye(non_absorbing.shape[0])
    diff_mat = identity_mat - non_absorbing
    det = np.linalg.det(diff_mat)
    return det != 0
