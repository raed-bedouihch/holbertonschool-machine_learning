#!/usr/bin/env python3
"""write a function that determines if you
should stop gradient descent early"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Returns: a boolean of whether the network
    should be stopped early, followed by the updated count"""
    stop = False
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1
    if count >= patience:
        stop = True
    return stop, count
