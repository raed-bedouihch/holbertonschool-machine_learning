#!/usr/bin/env python3
""" a class EncoderBlock that inherits from
tensorflow.keras.layers.Layer to create
an encoder block for a transformer"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """EncoderBlock class"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """constructor"""
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """Returns: a tensor of shape (batch, input_seq_len, dm)
        containing the block’s output"""
        output, weights = self.mha(x, x, x, mask)
        output = self.dropout1(output, training=training)
        output1 = self.layernorm1(x + output)
        dense_out = self.dense_hidden(output1)
        dense_out = self.dense_output(dense_out)
        dense_out = self.dropout2(dense_out, training=training)
        return self.layernorm2(output1 + dense_out)
