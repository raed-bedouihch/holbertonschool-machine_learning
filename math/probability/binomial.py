#!/usr/bin/env python3
"""a class Binomial that represents a Binomial distribution"""


class Binomial:
    """Binomial class"""

    def __init__(self, data=None, n=1, p=0.5):
        """constructor"""
        if data is None:
            if not isinstance(n, int) or n <= 0:
                raise ValueError('n must be a positive value')
            if not (0 < p < 1):
                raise ValueError('p must be greater than 0 and less than 1')
            self.n = n
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            mean = sum(data) / len(data)
            var = sum([(numb - mean) ** 2 for numb in data]) / len(data)
            p = 1 - (var / mean)
            n = round(mean / p)
            self.n = n
            self.p = float(mean / n)
            self.data = data

    def pmf(self, k):
        """Calculates the value of the PMF for a
        given number of “successes”"""
        if not isinstance(k, int):
            k = int(k)
        if k < 0 or k > self.n:
            return 0
        else:
            bin_coef = 1
            for i in range(1, k + 1):
                bin_coef *= (self.n - i + 1) / i
            return bin_coef * (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """Calculates the value of the CDF
        for a given number of “successes”"""
        if not isinstance(k, int):
            k = int(k)
        if k < 0 or k > self.n:
            return 0
        else:
            sum = 0
            for i in range(k + 1):
                sum += self.pmf(i)
            return sum
