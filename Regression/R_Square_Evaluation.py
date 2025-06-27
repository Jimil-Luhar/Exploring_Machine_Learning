import numpy as np

# Input data
housing_data = np.array([
    [1800, 3],    [2400, 4],    [1416, 2],    [3000, 5]
])
prices = np.array([350000, 475000, 230000, 640000])

# Step 1: Add a column of ones to include the bias/intercept term
bias_ones = np.ones(shape=(len(housing_data), 1))
X = np.append(bias_ones, housing_data, axis=1)  # Final design matrix (X)

# β = (XᵀX)^(-1) Xᵀy
coefficients_beta = np.linalg.inv(X.T @ X) @ X.T @ prices
print("Beta Coefficients (Intercept, Area Weight, Bedrooms Weight):")
print(coefficients_beta)

# Step 3: Predict prices using the learned model
predicted_prices = X @ coefficients_beta

# Calculate R² Score (Coefficient of Determination)
# R² = 1 - (SSE / SST)
# SST: Total Sum of Squares
# SSE: Sum of Squared Errors (Residuals)

SST = np.sum((prices - np.mean(prices)) ** 2)
SSE = np.sum((prices - predicted_prices) ** 2)
R_square = 1 - (SSE / SST)
print("R² Score:", R_square)

# Alternate (but same) way to calculate R² (in one single line)
R_alternate = 1 - (np.sum((prices - predicted_prices) ** 2) / np.sum((prices - np.mean(prices)) ** 2))
print("R² Score:", R_alternate) #R^2 : 0.997113020727609
