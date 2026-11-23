
import numpy as np

from simulate import simulate_returns
from methods import my_estimate_mu_sigma, my_min_variance


def report(name, manual, reference):
    manual = np.asarray(manual, dtype=float)
    reference = np.asarray(reference, dtype=float)

    max_error = np.max(np.abs(manual - reference))
    relative_error = (
        np.linalg.norm(manual - reference) /
        np.linalg.norm(reference)
    )

    print(
        f"{name:32s} "
        f"max error = {max_error:.2e}   "
        f"relative error = {relative_error:.2e}"
    )



R, mu, Sigma = simulate_returns(T=2000, seed=0)


m, S = my_estimate_mu_sigma(R)


w = my_min_variance(S)

# NumPy weights
w_lib = np.linalg.solve(S, np.ones(4))
w_lib = w_lib / w_lib.sum()

# True weights
w_true = np.linalg.solve(Sigma, np.ones(4))
w_true = w_true / w_true.sum()


print("2,000 OBSERVATIONS")

report("Weights vs NumPy", w, w_lib)
report("Weights vs truth (T=2000)", w, w_true)

print("\n Manual Minimum-Variance Weights:")
print(w)

print("\nSum of Weights:")
print(w.sum())

print("\nCovariance Matrix Positive Definite:")
print(np.linalg.eigvalsh(S).min() > 0)

print("\nCovariance Matrix Condition Number:")
print(np.linalg.cond(S))


#Same only 60 observations
R60, _, _ = simulate_returns(T=60, seed=0)

m60, S60 = my_estimate_mu_sigma(R60)
w60 = my_min_variance(S60)

print("\nSIMULATION: 60 OBSERVATIONS")

report("Weights vs truth (T=60)", w60, w_true)

print("\nMinimum-Variance Weights:")
print(w60)

print("\nSum of Weights:")
print(w60.sum())

print("\nCovariance Matrix Condition Number:")
print(np.linalg.cond(S60))
