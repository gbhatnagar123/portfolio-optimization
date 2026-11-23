
import numpy as np


def simulate_returns(T=2000, seed=0):
    rng = np.random.default_rng(seed)

    # True expected returns and volatilities
    mu = np.array([0.05, 0.08, 0.12, 0.07])
    vol = np.array([0.10, 0.15, 0.25, 0.12])

    # Correlation matrix
    Corr = np.array([
        [1.0, 0.3, 0.2, 0.1],
        [0.3, 1.0, 0.4, 0.2],
        [0.2, 0.4, 1.0, 0.3],
        [0.1, 0.2, 0.3, 1.0]
    ])

    # True covariance matrix
    Sigma = np.outer(vol, vol) * Corr

    # Generate returns
    R = rng.multivariate_normal(mu, Sigma, size=T)

    return R, mu, Sigma
