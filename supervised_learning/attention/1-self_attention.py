#!/usr/bin/env python3
""" a class SelfAttention that inherits from
tensorflow.keras.layers.Layer to calculate
the attention for machine translation based on this paper"""
import tensorflow as tf


class SelfAttention (tf.keras.layers.Layer):
    """SelfAttention  class"""

    def __init__(self, units):
        """constructor"""
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """Returns: context, weights"""
        expand_s_prev = tf.expand_dims(s_prev, 1)
        score = self.V(
            tf.nn.tanh(
                self.W(expand_s_prev) +
                self.U(hidden_states)))
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)
        return context, weights
