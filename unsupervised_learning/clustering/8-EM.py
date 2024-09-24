#!/usr/bin/env python3
"""
Expectation-Maximization algorithm for Gaussian Mixture Models (GMM)
"""
import numpy as np
from typing import Tuple, Optional
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X: np.ndarray, k: int, iterations: int = 1000, tol: float = 1e-5, verbose: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
 
    try:
        pi, m, S = initialize(X, k)
        prev_l = None

        for i in range(iterations):
            g, l = expectation(X, pi, m, S)
            
            # Stopping criterion
            if prev_l is not None and abs(l - prev_l) <= tol:
                break
            
            pi, m, S = maximization(X, g)
            
            # Verbose output for every 10 iterations
            if verbose and i % 10 == 0:
                print(f"Log Likelihood after {i} iterations: {l:.5f}")
            
            prev_l = l

        # Final verbose output
        if verbose:
            print(f"Log Likelihood after {i + 1} iterations: {l:.5f}")

        return pi, m, S, g, l

    except ValueError as ve:
        print(f"ValueError: {ve}")
    except TypeError as te:
        print(f"TypeError: {te}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return None, None, None, None, None
