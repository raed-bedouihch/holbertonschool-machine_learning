#!/usr/bin/env python3
"""a function that converts a gensim word2vec model
to a keras Embedding layer"""
import tensorflow as tf


def gensim_to_keras(model):
    """Returns: the trainable keras Embedding"""
    vocab_size, embedding_dims = model.wv.vectors.shape
    embedding_lay = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dims,
        weights=[
            model.wv.vectors],
        trainable=True)
    return embedding_lay
