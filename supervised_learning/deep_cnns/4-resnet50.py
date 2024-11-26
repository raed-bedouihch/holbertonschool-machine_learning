#!/usr/bin/env python3
"""a function  that builds the ResNet-50 architecture
as described in Deep Residual Learning for Image Recognition (2015)"""
from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Returns: the activated output of the projection block"""
    initializer = K.initializers.he_normal(seed=0)
    input_lay = K.Input(shape=(224, 224, 3))

    conv1 = K.layers.Conv2D(64, kernel_size=(7, 7), strides=(
        2, 2), padding='same', kernel_initializer=initializer)(input_lay)

    bn1 = K.layers.BatchNormalization(axis=3)(conv1)

    relu1 = K.layers.Activation('relu')(bn1)

    pool1 = K.layers.MaxPool2D(
        (3, 3), strides=(2, 2), padding='same')(relu1)

    proj1 = projection_block(pool1, [64, 64, 256], s=1)
    iden1 = identity_block(proj1, [64, 64, 256])
    iden2 = identity_block(iden1, [64, 64, 256])

    proj2 = projection_block(iden2, [128, 128, 512])
    iden3 = identity_block(proj2, [128, 128, 512])
    iden4 = identity_block(iden3, [128, 128, 512])
    iden5 = identity_block(iden4, [128, 128, 512])

    proj3 = projection_block(iden5, [256, 256, 1024])
    iden6 = identity_block(proj3, [256, 256, 1024])
    iden7 = identity_block(iden6, [256, 256, 1024])
    iden8 = identity_block(iden7, [256, 256, 1024])
    iden9 = identity_block(iden8, [256, 256, 1024])
    iden10 = identity_block(iden9, [256, 256, 1024])

    proj4 = projection_block(iden10, [512, 512, 2048])
    iden11 = identity_block(proj4, [512, 512, 2048])
    iden12 = identity_block(iden11, [512, 512, 2048])

    avg_pool = K.layers.AveragePooling2D((7, 7), padding='same')(iden12)
    output_lay = K.layers.Dense(
        units=1000,
        activation='softmax',
        kernel_initializer=initializer)(avg_pool)
    return K.Model(inputs=input_lay, outputs=output_lay)
