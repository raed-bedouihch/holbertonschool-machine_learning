#!/usr/bin/env python3
"""a function that calculates the scaled dot product attention"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """Returns: output, weights"""
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    matmul = tf.matmul(Q, K, transpose_b=True)
    scale = matmul / tf.math.sqrt(dk)
    if mask is not None:
        scale += (mask * -1e9)
    weights = tf.nn.softmax(scale, axis=-1)
    output = tf.matmul(weights, V)
    return output, weights
