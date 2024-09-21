#!/usr/bin/env python3
""" 3. Initialize Exponential """
""" 4. Exponential PDF """


class Exponential:
            if len(data) <= 1:
                raise ValueError("data must contain multiple values")
            self.lambtha = float((1 / sum(data)) / (1 / len(data)))
    def pdf(self, x):
        """Calculates the value of the PDF for a given time period"""
        e = 2.7182818285
        if x < 0:
            return (0)
        return self.lambtha * e ** ((-1 * self.lambtha) * x)