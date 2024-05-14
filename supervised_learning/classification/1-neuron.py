#!/usr/bin/env python3
""" 1. Neuron """
import numpy as np


class Neuron:
    """ Class that defines a single neuron performing binary classification """

    def __init__(self, nx):
        """ Class constructor for Neuron """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ Returns the values of the weights vector """
        return self.__W

    @property
    def b(self):
        """ Returns the value of the bias neuron """
        return self.__b

    @property
    def A(self):
        """ Returns the value of the activation function output """
        return self.__A
