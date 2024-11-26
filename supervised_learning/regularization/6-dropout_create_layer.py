#!/usr/bin/env python3
"""write a function that creates a layer of a neural network using dropout"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """Returns: the output of the new layer"""
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0, mode=("fan_avg"))
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer)(prev)
    if training:
        dropout_layer = tf.keras.layers.Dropout(
            rate=1 - keep_prob)(layer, training=True)
    return dropout_layer
