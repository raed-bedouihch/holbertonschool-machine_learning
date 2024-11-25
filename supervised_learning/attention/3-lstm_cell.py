#!/usr/bin/env python3
"""Create a class LSTMCell that represents an LSTM unit"""
import numpy as np


class LSTMCell:
    """LSTMCell class"""

    def __init__(self, i, h, o):
        """constructor"""
        self.bf = np.zeros((1, h))
        self.Wf = np.random.normal(size=(h + i, h))

        self.Wu = np.random.normal(size=(h + i, h))
        self.bu = np.zeros((1, h))

        self.Wc = np.random.normal(size=(h + i, h))
        self.bc = np.zeros((1, h))

        self.Wo = np.random.normal(size=(h + i, h))
        self.bo = np.zeros((1, h))

        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """performs forward propagation for one time step"""
        concat = np.concatenate((h_prev, x_t), axis=1)
        f = 1 / (1 + np.exp(- (np.matmul(concat, self.Wf) + self.bf)))
        u = 1 / (1 + np.exp(- (np.matmul(concat, self.Wu) + self.bu)))
        c = np.tanh(np.matmul(concat, self.Wc) + self.bc)
        c_next = c_prev * f + u * c
        o = 1 / (1 + np.exp(- (np.matmul(concat, self.Wo) + self.bo)))
        h_next = o * np.tanh(c_next)
        x = np.matmul(h_next, self.Wy) + self.by
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        y = e_x / e_x.sum(axis=1, keepdims=True)
        return h_next, c_next, y
