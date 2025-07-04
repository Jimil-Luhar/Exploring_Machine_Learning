import numpy as np

# Input: housing data
house_size = np.array([[1200], [1500], [1000]])  # Feature matrix (3x1)
prices = np.array([240000, 300000, 200000])      # Target vector (3,)
theta = np.array([100000, 100])                  # Initial guess: [intercept, slope]

# Hyperparameters
learning_rate = 0.01
m = len(prices)

# Design matrix with bias term
X_b = np.c_[np.ones((m, 1)), house_size]         # Shape: (3, 2)
predictions = X_b.dot(theta)                     # Shape: (3,)
gradient = (1/m) * X_b.T.dot(predictions - prices)  # Shape: (2,)
theta = theta - learning_rate * gradient
print(f"Updated parameters: {theta}")
