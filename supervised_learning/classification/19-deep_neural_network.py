#!/usr/bin/env python3
"""DeepNeuralNetwork"""
import numpy as np


class DeepNeuralNetwork:
    """ defines a deep neural network performing binary classification """

    def __init__(self, nx, layers):
        """constructor for DeepNeuralNetwork"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for layer in range(self.L):
            if not isinstance(layers[layer], int) or layers[layer] < 1:
                raise TypeError("layers must be a list of positive integers")
            if layer == 0:
                he_initialized_weights = np.random.randn(
                    layers[layer], nx) * np.sqrt(2 / nx)
            else:
                he_initialized_weights = np.random.randn(
                    layers[layer], layers[layer - 1]
                ) * np.sqrt(2 / layers[layer - 1])
            self.__weights['W' + str(layer + 1)] = he_initialized_weights
            self.__weights['b' + str(layer + 1)] = np.zeros((layers[layer], 1))

    @property
    def L(self):
        """getter for L"""
        return self.__L

    @property
    def cache(self):
        """getter for cache"""
        return self.__cache

    @property
    def weights(self):
        """getter for weights"""
        return self.__weights

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network."""
        self.__cache['A0'] = X  # Save input data in cache
        A = X  # Start with the input as the output for the first layer

        for layer in range(1, self.L + 1):
            W = self.__weights['W' + str(layer)]
            b = self.__weights['b' + str(layer)]

            # Calculate Z for the current layer
            Z = np.dot(W, A) + b

            # Apply the sigmoid activation function
            A = 1 / (1 + np.exp(-Z))

            # Save the activated output in the cache
            self.__cache['A' + str(layer)] = A

        return A, self.__cache

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression."""
        m = Y.shape[1]  # Number of examples

        # Calculate the cost
        cost = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        cost = -np.sum(cost) / m  # Average cost

        return cost
