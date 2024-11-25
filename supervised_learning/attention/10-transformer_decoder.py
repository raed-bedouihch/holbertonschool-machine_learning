#!/usr/bin/env python3
"""a class Decoder that inherits from
tensorflow.keras.layers.Layer to create
the decoder for a transformer"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """Decoder class"""

    def __init__(
            self,
            N,
            dm,
            h,
            hidden,
            target_vocab,
            max_seq_len,
            drop_rate=0.1):
        """constructor"""
        super(Decoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for i in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """Returns: a tensor of shape (batch, target_seq_len, dm)
        containing the decoder output"""
        input_seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:input_seq_len, :]
        x = self.dropout(x, training=training)
        for block in self.blocks:
            x = block(
                x,
                encoder_output,
                training,
                look_ahead_mask,
                padding_mask)
        return x
