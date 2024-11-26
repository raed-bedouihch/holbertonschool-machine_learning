#!/usr/bin/env python3
"""a function that calculates the determinant of a matrix"""


def determinant(matrix):
    """Returns: the determinant of matrix"""

    if not isinstance(matrix, list) or not all(
        isinstance(
            row,
            list) for row in matrix):
        raise TypeError('matrix must be a list of lists')
    if matrix == []:
        raise TypeError('matrix must be a list of lists')
    n_rows = len(matrix)
    if n_rows == 1 and len(matrix[0]) == 0:
        return 1
    if any(len(row) != n_rows for row in matrix) or n_rows == 0:
        raise ValueError('matrix must be a square matrix')
    if n_rows == 1:
        return matrix[0][0]
    if n_rows == 2:
        return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]
    det = 0
    for col in range(n_rows):
        minor = []
        for row in range(1, n_rows):
            minor_row = matrix[row][:col] + matrix[row][col + 1:]
            minor.append(minor_row)
        cofactor = ((-1) ** col) * determinant(minor) * matrix[0][col]
        det += cofactor
    return det
