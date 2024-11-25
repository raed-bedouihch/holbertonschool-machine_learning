#!/usr/bin/env python3
"""a class MultiHeadAttention that inherits from
tensorflow.keras.layers.Layer to perform multi head attention"""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """MultiHeadAttention class"""

    def __init__(self, dm, h):
        """constructor"""
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def call(self, Q, K, V, mask):
        """Returns: output, weights"""
        batch = tf.shape(Q)[0]
        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)
        Q = tf.reshape(Q, (batch, -1, self.h, self.depth))
        K = tf.reshape(K, (batch, -1, self.h, self.depth))
        V = tf.reshape(V, (batch, -1, self.h, self.depth))
        Q = tf.transpose(Q, perm=[0, 2, 1, 3])
        K = tf.transpose(K, perm=[0, 2, 1, 3])
        V = tf.transpose(V, perm=[0, 2, 1, 3])
        output, weights = sdp_attention(Q, K, V, mask)
        scaled = tf.transpose(output, perm=[0, 2, 1, 3])
        concat = tf.reshape(scaled, (batch, -1, self.dm))
        return self.linear(concat), weights
