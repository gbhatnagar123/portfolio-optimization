# Portfolio Optimization

## Overview
This project explores portfolio optimization using covariance matrices and Modern Portfolio Theory. The main objective is to calculate the Global Minimum Variance Portfolio (GMVP), implement the optimization process using custom numerical methods, and understand the relationship between portfolio risk and expected return.

## Dataset
The project uses the **10 Industry Portfolios** dataset from the [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).

- 10 industry portfolios
- 1,203 monthly observations
- Time period: July 1926 – September 2026
- Monthly returns converted from percentages to decimals

The estimated monthly expected returns and covariance matrix are annualized by multiplying each by 12.

## Methods

### Mathematical Foundations
- Expected portfolio return and variance
- Covariance matrix properties
- Minimum-variance portfolio derivation using Lagrange multipliers
- Efficient frontier derivation
- Portfolio risk and return analysis

### Python Implementation
- Custom Gaussian elimination with partial pivoting
- Sample mean and covariance matrix estimation
- Minimum-variance portfolio weight calculation
- Portfolio weight normalization
- Verification against `numpy.linalg.solve`

### Simulation and Testing
- Four-asset multivariate normal return simulation
- Comparison of estimated portfolio weights against theoretical weights
- Solver verification using NumPy
- Testing with different sample sizes (2,000 and 60 observations)
- Covariance matrix condition number and numerical stability analysis

### Real-Data Analysis
- Minimum-variance portfolio using 10 industry portfolios
- Annualized expected returns and covariance matrix
- Covariance matrix condition number analysis
- Visulaization of the Efficent Frontier 

## Project Structure

- `data/`: Dataset used for analysis
- `src/`: Python implementation and simulation scripts
- `notebooks/`: Jupyter notebooks and experiments
- `figures/`: Generated plots and visualizations
- `docs/`: Project report and planning documents

## Technologies
- Python
- NumPy
- Pandas
- Matplotlib
- Git and GitHub