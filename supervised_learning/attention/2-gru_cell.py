#!/usr/bin/env python3
"""Create a class GRUCell that represents a gated recurrent unit"""
import numpy as np


class GRUCell:
    """GRUCell class"""

    def __init__(self, i, h, o):
        """constructor"""
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.Wz = np.random.normal(size=(h + i, h))
        self.Wr = np.random.normal(size=(h + i, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))
        self.Wh = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))

    def forward(self, h_prev, x_t):
        """performs forward propagation for one time step"""
        concat = np.concatenate((h_prev, x_t), axis=1)
        z = 1 / (1 + np.exp(- (np.matmul(concat, self.Wz) + self.bz)))
        r = 1 / (1 + np.exp(- (np.matmul(concat, self.Wr) + self.br)))

        concat2 = np.concatenate((h_prev * r, x_t), axis=1)
        h_next = np.tanh(np.matmul(concat2, self.Wh) + self.bh)
        h_next *= z
        h_next += (1 - z) * h_prev
        x = np.matmul(h_next, self.Wy) + self.by
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        y = e_x / e_x.sum(axis=1, keepdims=True)
        return h_next, y
