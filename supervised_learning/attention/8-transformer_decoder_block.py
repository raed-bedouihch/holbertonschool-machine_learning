#!/usr/bin/env python3
""" a class DecoderBlock that inherits from
tensorflow.keras.layers.Layer to create
an decoder block for a transformer"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """DecoderBlock class"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """constructor"""
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """Returns: a tensor of shape (batch, target_seq_len, dm)
        containing the block’s output"""
        output, weights = self.mha1(x, x, x, look_ahead_mask)
        output = self.dropout1(output, training=training)
        out1 = self.layernorm1(x + output)

        output2, weights2 = self.mha2(
            out1, encoder_output, encoder_output, padding_mask)
        output2 = self.dropout2(output2, training=training)
        out2 = self.layernorm2(out1 + output2)

        dense_out = self.dense_hidden(out2)
        dense_out = self.dense_output(dense_out)
        dense_out = self.dropout3(dense_out, training=training)
        return self.layernorm3(out2 + dense_out)
