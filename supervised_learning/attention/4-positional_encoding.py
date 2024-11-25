#!/usr/bin/env python3
"""a function that calculates the positional encoding for a transformer"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """Returns: a numpy.ndarray of shape (max_seq_len, dm)
    containing the positional encoding vectors"""
    pos_enc = np.zeros((max_seq_len, dm))
    for j in range(max_seq_len):
        for i in range(dm):
            angle_rate = j / np.power(10000, (2 * (i // 2) / dm))
            if i % 2 == 0:
                pos_enc[j, i] = np.sin(angle_rate)
            else:
                pos_enc[j, i] = np.cos(angle_rate)
    return pos_enc
