#!/usr/bin/env python3
"""a function that calculates the inverse of a matrix"""

adjugate = __import__('3-adjugate').adjugate
determinant = __import__('0-determinant').determinant


def inverse(matrix):
    """Returns: the inverse of matrix or None if matrix is singular"""

    if not isinstance(matrix, list) or not all(
        isinstance(
            row,
            list) for row in matrix):
        raise TypeError('matrix must be a list of lists')
    if matrix == []:
        raise TypeError('matrix must be a list of lists')
    n_rows = len(matrix)
    if any(len(row) != n_rows for row in matrix) or n_rows == 0:
        raise ValueError('matrix must be a non-empty square matrix')
    adj_matrix = adjugate(matrix)
    n_rows = len(adj_matrix)
    det = determinant(matrix)
    if det == 0:
        return None
    for row in range(n_rows):
        for col in range(n_rows):
            adj_matrix[row][col] /= det
    return adj_matrix
