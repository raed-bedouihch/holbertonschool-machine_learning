#!/usr/bin/env python3
"""a function that calculates the minor matrix of a matrix"""

determinant = __import__('0-determinant').determinant


def minor(matrix):
    """Returns: the minor matrix of matrix"""

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
    if n_rows == 1:
        return [[1]]
    minor_mat = []
    for row in range(n_rows):
        minor_row = []
        for col in range(n_rows):
            sub_matrix = []
            for r in range(n_rows):
                if r != row:
                    minor_sub = matrix[r][:col] + matrix[r][col + 1:]
                    sub_matrix.append(minor_sub)
            minor_row.append(determinant(sub_matrix))
        minor_mat.append(minor_row)
    return minor_mat
