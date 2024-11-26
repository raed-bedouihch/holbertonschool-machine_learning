#!/usr/bin/env python3
"""a function that builds the DenseNet-121
architecture as described in Densely Connected
Convolutional Networks"""
from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """Returns: the keras model"""
    initializer = K.initializers.he_normal(seed=0)
    input_lay = K.Input(shape=(224, 224, 3))

    bn1 = K.layers.BatchNormalization(axis=3)(input_lay)
    relu1 = K.layers.Activation('relu')(bn1)
    conv1 = K.layers.Conv2D(filters=64,
                            kernel_size=(7, 7), strides=(2, 2), padding='same',
                            kernel_initializer=initializer)(relu1)
    pool1 = K.layers.MaxPool2D(
        (3, 3), strides=(2, 2), padding='same')(conv1)

    dense1, nb_filters = dense_block(pool1, 64, growth_rate, 6)
    trans1, nb_filters = transition_layer(dense1, nb_filters, compression)

    dense2, nb_filters = dense_block(trans1, nb_filters, growth_rate, 12)
    trans2, nb_filters = transition_layer(dense2, nb_filters, compression)

    dense3, nb_filters = dense_block(trans2, nb_filters, growth_rate, 24)
    trans3, nb_filters = transition_layer(dense3, nb_filters, compression)

    dense4, nb_filters = dense_block(trans3, nb_filters, growth_rate, 16)

    avg_pool = K.layers.AveragePooling2D((7, 7), strides=1)(dense4)
    output_lay = K.layers.Dense(
        units=1000,
        activation='softmax',
        kernel_initializer=initializer)(avg_pool)
    return K.Model(inputs=input_lay, outputs=output_lay)
