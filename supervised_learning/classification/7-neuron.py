#!/usr/bin/env python3
""" 1. Neuron """
import numpy as np
import matplotlib.pyplot as plt


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

    def forward_prop(self, X):
        """ Calculates the forward propagation of the neuron """
        z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A

    def cost(self, Y, A):
        """ Calculates the cost of the model using logistic regression """
        m = Y.shape[1]
        cost = -np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A))) / m
        return cost

    def evaluate(self, X, Y):
        """ Evaluates the neuron’s predictions """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """ Calculates one pass of gradient descent on the neuron """
        m = Y.shape[1]
        dz = A - Y
        dw = np.matmul(X, dz.T) / m
        db = np.sum(dz) / m
        self.__W -= (alpha * dw).T
        self.__b -= alpha * db
        return self.__W, self.__b

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """ Trains the neuron V2 """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step < 1 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        if graph:
            import matplotlib.pyplot as plt
            x = np.arange(0, iterations + 1, step)
            y = np.empty((iterations // step) + 1, dtype=float)
        for i in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)
            if verbose and i % step == 0:
                cost = self.cost(Y, A)
                print("Cost after {} iterations: {}".format(i, cost))
            if graph and i % step == 0:
                y[i // step] = cost
        if verbose:
            cost = self.cost(Y, A)
            print("Cost after {} iterations: {}".format(iterations, cost))
        if graph:
            y[-1] = cost
            plt.plot(x, y, 'b-')
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()
        return self.evaluate(X, Y)
