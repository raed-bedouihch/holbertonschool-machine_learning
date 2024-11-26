#!/usr/bin/env python3
"""a function that builds a generator and discriminator"""
from tensorflow import keras


def convolutional_GenDiscr():
    """Returns: the concatenated output of the generator and discriminator"""

    def get_generator():
        """generator model"""
        # generator model
        inp_lay = keras.layers.Input(shape=(16, ))
        x = keras.layers.Dense(2048)(inp_lay)
        x = keras.layers.Reshape((2, 2, 512))(x)
        x = keras.layers.UpSampling2D()(x)
        x = keras.layers.Conv2D(64, (3, 3), padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu', name='activation_1')(x)
        x = keras.layers.UpSampling2D()(x)
        x = keras.layers.Conv2D(16, (3, 3), padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu', name='activation_2')(x)
        x = keras.layers.UpSampling2D()(x)
        x = keras.layers.Conv2D(1, (3, 3), padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        out_lay = keras.layers.Activation('tanh', name='activation_3')(x)
        return keras.models.Model(inp_lay, out_lay, name='generator')

    def get_discriminator():
        """discriminator model"""
        # discriminator model
        inp_lay = keras.layers.Input(shape=(16, 16, 1))
        x = keras.layers.Conv2D(32, (3, 3), padding='same')(inp_lay)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Activation('relu', name='activation_4')(x)
        x = keras.layers.Conv2D(64, (3, 3), padding='same')(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Activation('relu', name='activation_5')(x)
        x = keras.layers.Conv2D(128, (3, 3), padding='same')(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Activation('relu', name='activation_6')(x)

        x = keras.layers.Conv2D(256, (3, 3), padding='same')(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Activation('relu', name='activation_7')(x)
        x = keras.layers.Flatten()(x)
        out_lay = keras.layers.Dense(1)(x)
        return keras.models.Model(inp_lay, out_lay, name='discriminator')

    return get_generator(), get_discriminator()
