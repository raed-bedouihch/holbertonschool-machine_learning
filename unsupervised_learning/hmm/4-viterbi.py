#!/usr/bin/env python3
"""a function that calculates the most likely
sequence of hidden states for a hidden markov model"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """Returns: path, P, or None, None on failure"""
    try:
        T = Observation.shape[0]
        N = Emission.shape[0]
        V = np.zeros((N, T))
        path = np.zeros((N, T), dtype=int)
        V[:, 0] = Initial.flatten() * Emission[:, Observation[0]]
        for t in range(1, T):
            for n in range(N):
                prob = V[:, t - 1] * Transition[:, n] * \
                    Emission[n, Observation[t]]
                V[n, t] = np.max(prob)
                path[n, t] = np.argmax(prob)
        P = np.max(V[:, T - 1])
        best_state = np.argmax(V[:, T - 1])
        best_path = [best_state]
        for t in range(T - 1, 0, -1):
            best_state = path[best_state, t]
            best_path.append(best_state)
        best_path.reverse()
        return best_path, P
    except BaseException:
        return None, None
