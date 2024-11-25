#!/usr/bin/env python3
"""  a class RNNEncoder that inherits from
tensorflow.keras.layers.Layer to encode
for machine translation"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """RNNEncoder class"""

    def __init__(self, vocab, embedding, units, batch):
        """constructor"""
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            recurrent_initializer='glorot_uniform',
            return_sequences=True,
            return_state=True)

    def initialize_hidden_state(self):
        """Initializes the hidden states for the RNN
        cell to a tensor of zeros"""
        return tf.zeros(shape=(self.batch, self.units))

    def call(self, x, initial):
        """Returns: outputs, hidden"""
        embed = self.embedding(x)
        outputs, hidden = self.gru(embed, initial_state=initial)
        return outputs, hidden
