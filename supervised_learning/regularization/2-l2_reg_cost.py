#!/usr/bin/env python3
"""write a function that calculates the
cost of a neural network with L2 regularization"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """Returns: a tensor containing the total
    cost for each layer of the network,
    accounting for L2 regularization"""
    reg_loss = []
    for layer in model.layers:
        reg_loss.append(tf.reduce_sum(layer.losses) + cost)
    return tf.stack(reg_loss[1:])
