#!/usr/bin/env python3
"""a function that computes the policy with a weight of a matrix."""
import numpy as np


def policy(matrix, weight):
    """returns the computed policy"""
    z = np.dot(matrix, weight)
    e_z = np.exp(z - np.max(z))
    return e_z / e_z.sum()


def policy_gradient(state, weight):
    """computes the Monte-Carlo policy gradient based on
    a state and a weight matrix."""
    probs = policy(state, weight)
    action = np.random.choice(len(probs), p=probs)
    grad = -probs
    grad[action] += 1
    gradient = np.outer(state, grad)
    return action, gradient
