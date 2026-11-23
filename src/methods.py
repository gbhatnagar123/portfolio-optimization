import numpy as np


# Solve Ax = b using Gaussian elimination with partial pivoting.
def my_gaussian_elimination(A, b, return_steps=False):
    A = np.array(A, dtype=float).copy()
    b = np.array(b, dtype=float).copy()

    n = len(b)
    augmented = np.column_stack((A, b))
    steps = [("Initial augmented matrix", augmented.copy())]

    # Forward elimination
    for k in range(n):
        # Find the row with the largest absolute pivot
        pivot_row = k + np.argmax(np.abs(A[k:, k]))

        # Swap rows if needed
        if pivot_row != k:
            A[[k, pivot_row]] = A[[pivot_row, k]]
            b[[k, pivot_row]] = b[[pivot_row, k]]
            steps.append((
                f"Swap rows {k + 1} and {pivot_row + 1}",
                np.column_stack((A, b)).copy()
            ))

        if abs(A[k, k]) < 1e-12:
            raise ValueError("Matrix is singular or nearly singular.")

        # Eliminate entries below the pivot
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] = A[i, k:] - factor * A[k, k:]
            b[i] = b[i] - factor * b[k]

            steps.append((
                f"Eliminate row {i + 1}, using row {k + 1}",
                np.column_stack((A, b)).copy()
            ))

    # Back substitution
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i]

    steps.append(("Solution x", x.copy()))

    if return_steps:
        return x, steps

    return x


# Estimate the sample mean and sample covariance from R.
def my_estimate_mu_sigma(R):
    T = R.shape[0]

    mu = R.mean(axis=0)
    centered_R = R - mu
    Sigma = (centered_R.T @ centered_R) / (T - 1)

    return mu, Sigma


# Minimum-variance weights
# x = solve(Sigma, 1), then w = x / sum(x).
def my_min_variance(Sigma):
    n = Sigma.shape[0]
    one = np.ones(n)

    x = my_gaussian_elimination(Sigma, one)
    w = x / np.sum(x)

    return w
