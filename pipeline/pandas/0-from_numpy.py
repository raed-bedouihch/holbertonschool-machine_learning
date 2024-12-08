#!/usr/bin/env python3
"""a function that that creates a pd.DataFrame from a np.ndarray"""
import pandas as pd


def from_numpy(array):
    """Returns: the newly created pd.DataFrame"""
    n_cols = array.shape[1]
    if n_cols <= 26:
        cols = [chr(i) for i in range(65, 65 + n_cols)]
    return pd.DataFrame(array, columns=cols)
