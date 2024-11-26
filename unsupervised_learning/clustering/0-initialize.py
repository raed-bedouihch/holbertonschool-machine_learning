#!/usr/bin/env python3
"""a function that initializes cluster centroids for K-means"""
import numpy as np


def initialize(X, k):
    """Returns: a numpy.ndarray of shape (k, d)
    containing the initialized centroids for each cluster,
    or None on failure"""
    try:
        n, d = X.shape
        if not isinstance(k, int) or k <= 0:
            return None
        centroids = np.random.uniform(
            low=np.min(
                X, axis=0), high=np.max(
                X, axis=0), size=(
                    k, d))
        return centroids
    except Exception:
        return None
