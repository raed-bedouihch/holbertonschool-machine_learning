def summation_i_squared(n):
    # Check if n is a valid number
    if not isinstance(n, int) or n < 1:
        return None

    # Calculate the sum using the formula
    return n * (n + 1) * (2 * n + 1) // 6
