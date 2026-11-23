import numpy as np

from methods import my_gaussian_elimination, my_min_variance


# Two-asset example
mu = np.array([0.08, 0.12])

Sigma = np.array([
    [4.0, 1.0],
    [1.0, 9.0]
])

one = np.ones(2)

print("Expected Return Vector:")
print(mu)

print("\nCovariance Matrix:")
print(Sigma)


# Solve Sigma x = 1 using Gaussian elimination
x, steps = my_gaussian_elimination(Sigma, one, return_steps=True)

print("\nGaussian Elimination Steps:")

for description, result in steps:
    print(f"\n{description}:")
    print(result)


# Normalize x to obtain portfolio weights
w = my_min_variance(Sigma)

print("\nUnnormalized Solution x:")
print(x)

print("\nSum of x:")
print(np.sum(x))

print("\nMinimum Variance Weights:")
print(w)

print("\nSum of Portfolio Weights:")
print(np.sum(w))


# Calculate expected portfolio return and variance
portfolio_return = w @ mu
portfolio_variance = w @ Sigma @ w

print("\nExpected Portfolio Return:")
print(portfolio_return)

print("\nPortfolio Variance:")
print(portfolio_variance)


# Verification using NumPy
x_lib = np.linalg.solve(Sigma, one)
w_lib = x_lib / np.sum(x_lib)

print("\nNumPy Verification Weights:")
print(w_lib)

print("\nMaximum Absolute Difference:")
print(np.max(np.abs(w - w_lib)))
