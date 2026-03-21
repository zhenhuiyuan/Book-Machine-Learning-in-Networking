#Copyright@Zhenhui Yuan, 2025

import numpy as np
import pandas as pd
from scipy.stats import beta, norm

# Dataset with missing values
data = {
    "Packet Loss": [0.5, np.nan, 0.2, 0.7],  # Modeled with Beta distribution
    "Delay": [20, 18, np.nan, 22]  # Modeled with Gaussian distribution
}

# Convert to a DataFrame
df = pd.DataFrame(data)

# Display the original dataset
print("Original Dataset with Missing Values:")
print(df)


# Function to fit a distribution and suggest values for missing data
def fit_and_impute(data, distribution, fit_args=None, n_samples=3):
    """
    Fit a distribution to observed data and suggest values for missing data.

    Parameters:
    - data: pandas Series with missing values
    - distribution: scipy.stats distribution object
    - fit_args: additional arguments for fitting, if any
    - n_samples: number of samples to suggest for missing values

    Returns:
    - params: estimated parameters of the distribution
    - suggestions: list of suggested values for each missing value
    """
    # Separate observed values
    observed = data.dropna()

    # Fit the distribution using MLE
    if fit_args:
        params = distribution.fit(observed, **fit_args)
    else:
        params = distribution.fit(observed)

    # Sample n_samples values for each missing value
    missing_count = data.isna().sum()
    suggestions = distribution.rvs(*params[:-2], loc=params[-2], scale=params[-1], size=(missing_count, n_samples))

    return params, suggestions


# Fit Beta distribution to Packet Loss
beta_params, beta_suggestions = fit_and_impute(
    df["Packet Loss"], beta, fit_args={"floc": 0, "fscale": 1}, n_samples=3
)

# Fit Gaussian distribution to Delay
norm_params, norm_suggestions = fit_and_impute(
    df["Delay"], norm, n_samples=3
)

# Print the estimated parameters
print("\nEstimated Parameters:")
print(
    f"Packet Loss (Beta): a={beta_params[0]:.3f}, b={beta_params[1]:.3f}, loc={beta_params[2]:.3f}, scale={beta_params[3]:.3f}")
print(f"Delay (Gaussian): mean={norm_params[0]:.3f}, std={norm_params[1]:.3f}")

# Print suggestions for missing values
print("\nSuggested Values for Missing Data:")
print(f"Packet Loss Suggestions: {beta_suggestions.tolist()}")
print(f"Delay Suggestions: {norm_suggestions.tolist()}")
