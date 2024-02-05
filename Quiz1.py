import numpy as np


def free_num(scalar, free_numbers, scalar_index, index, matrix):
    free_numbers[index] = (free_numbers[index] + scalar * free_numbers[scalar_index]) / matrix[0,0]
    return free_numbers


def row_addition_elementary_matrix(n, target_row, source_row, scalar=1.0):
    if target_row < 0 or source_row < 0 or target_row >= n or source_row >= n:
        raise ValueError("Invalid row indices.")
    if target_row == source_row:
        raise ValueError("Source and target rows cannot be the same.")
    elementary_matrix = np.identity(n)
    elementary_matrix[target_row, source_row] = scalar
    return elementary_matrix


def scalar_multiplication_elementary_matrix(n, row_index, scalar):
    if row_index < 0 or row_index >= n:
        raise ValueError("Invalid row index.")
    if scalar == 0:
        raise ValueError("Scalar cannot be zero for row multiplication.")
    elementary_matrix = np.identity(n)
    elementary_matrix[row_index, row_index] = scalar
    return elementary_matrix


def inverse(matrix, free_numbers):
    print(
        f"=================== Finding the inverse of a non-singular matrix using elementary row operations ===================\n {matrix}\n")

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square.")

    n = matrix.shape[0]
    identity = np.identity(n)

    # Perform row operations to transform the input matrix into the identity matrix
    for i in range(n):
        if matrix[i, i] == 0:
            raise ValueError("Matrix is singular, cannot find its inverse.")

        if matrix[i, i] != 1:
            # Scale the current row to make the diagonal element 1
            scalar = 1.0 / matrix[i, i]
            elementary_matrix = scalar_multiplication_elementary_matrix(n, i, scalar)
            print(f"elementary matrix to make the diagonal element 1 :\n {elementary_matrix} \n")
            matrix = np.dot(elementary_matrix, matrix)
            print(f"The matrix after elementary operation :\n {matrix}")
            print(
                "------------------------------------------------------------------------------------------------------------------")
            identity = np.dot(elementary_matrix, identity)

        # Zero out the elements above and below the diagonal
        for j in range(n):
            if i != j:
                scalar = -matrix[j, i]
                elementary_matrix = row_addition_elementary_matrix(n, j, i, scalar)
                print(f"elementary matrix for R{j + 1} = R{j + 1} + ({scalar}R{i + 1}):\n {elementary_matrix} \n")
                matrix = np.dot(elementary_matrix, matrix)
                free_numbers = free_num(scalar, free_numbers, i, j, matrix)
                print(f"The matrix after elementary operation :\n {matrix}")
                print(
                    "------------------------------------------------------------------------------------------------------------------")

                identity = np.dot(elementary_matrix, identity)

    for i in range(len(free_numbers)):
        print(f"Variable number {i + 1} = {free_numbers[i]}")

    return identity,


if __name__ == '__main__':
    A = np.array([[-1, -3, 4],
                  [0, 1, -2],
                  [2, -2, 3]])

    try:
        free_n = np.array([1.0, 1.0, 1.0])
        A_inverse = inverse(A, free_n)
        print("\nInverse of matrix A: \n")
        print(
            "=====================================================================================================================")
        print(A_inverse)

    except ValueError as e:
        print(str(e))
