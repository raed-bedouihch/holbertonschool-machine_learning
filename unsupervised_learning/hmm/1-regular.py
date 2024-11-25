#!/usr/bin/env python3
"""a function that determines the steady state probabilities of a regular markov chain"""
import numpy as np


def is_regular(P):
    """Checks if a matrix P is regular (all elements > 0 after some power)"""
    n = P.shape[0]
    power = P.copy()
    for i in range(n):
        power = np.dot(power, P)
        return np.all(power > 0)


def regular(P):
    """Returns: a numpy.ndarray of shape (1, n)
    containing the steady state probabilities,
    or None on failure"""
    if not isinstance(
            P,
            np.ndarray) or P.shape[0] != P.shape[1] or not is_regular(P):
        return None
    n = P.shape[0]
    if not np.allclose(np.sum(P, axis=1), 1):
        return None
    transition_mat = np.transpose(P) - np.eye(n)
    transition_mat = np.vstack([transition_mat, np.ones(n)])
    steady_state_vec = np.zeros(n)
    steady_state_vec = np.append(steady_state_vec, 1)
    steady_state_vec[-1] = 1
    try:
        steady_state_probs = np.linalg.lstsq(
            transition_mat, steady_state_vec, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    return np.array([steady_state_probs])
