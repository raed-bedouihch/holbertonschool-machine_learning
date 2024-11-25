#!/usr/bin/env python3
"""a class RNNDecoder that inherits from
tensorflow.keras.layers.Layer to decode
for machine translation"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """RNNDecoder class"""

    def __init__(self, vocab, embedding, units, batch):
        """constructor"""
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            recurrent_initializer='glorot_uniform',
            return_sequences=True,
            return_state=True)
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """Returns: y, s"""
        units = s_prev.shape[1]
        attention = SelfAttention(units)
        context, weights = attention(s_prev, hidden_states)
        embed = self.embedding(x)
        concat = tf.concat([tf.expand_dims(context, 1), embed], axis=-1)
        outputs, hidden = self.gru(concat)
        outputs = tf.reshape(outputs, (outputs.shape[0], outputs.shape[2]))
        return self.F(outputs), hidden
