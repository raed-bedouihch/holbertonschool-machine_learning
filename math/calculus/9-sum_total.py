#!/usr/bin/env python3
"""a function that calculates the sum of i^2 from i =1 to n"""


def summation_i_squared(n):
    """Return the integer value of the sum"""
    if not isinstance(n, int) or n < 1:
        return None
    else:
        return (n * (n + 1) * (2 * n + 1)) // 6
