#!/usr/bin/env python3
"""a function that creates all masks for training/validation"""
import tensorflow as tf


def create_masks(inputs, target):
    """Returns: encoder_mask, combined_mask, decoder_mask"""
    def padding_mask(x):
        """create padding mask for a tensor"""
        mask = tf.cast(tf.math.equal(x, 0), tf.float32)
        return mask[:, tf.newaxis, tf.newaxis, :]

    encoder_mask = padding_mask(inputs)
    decoder_padding_mask = padding_mask(target)
    seq_len = tf.shape(target)[1]
    look_ahead_mask = 1 - \
        tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    combined_mask = tf.maximum(decoder_padding_mask, look_ahead_mask)
    decoder_mask = padding_mask(inputs)
    return encoder_mask, combined_mask, decoder_mask
