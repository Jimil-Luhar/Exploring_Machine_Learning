import numpy as np

# Define the coefficients of the linear regression model.
# b0 = intercept, b1 = coefficient for area (in square meters),
# b2 = coefficient for age (in years), b3 = coefficient for number of bathrooms
coefficients = np.array([50000, 3000, -2000, 15000])

# Define the feature values for a new house we want to predict the price of.
# Format: [1 (intercept term), area=150 sqm, age=10 years, 2 bathrooms]
new_house = np.array([1, 150, 10, 2])

# Compute the predicted price using the dot product of coefficients and house features
# This implements: price = b0*1 + b1*area + b2*age + b3*number_of_bathrooms
predicted_price = np.dot(coefficients, new_house)

# Display the predicted house price, formatted with commas and two decimal places
print(f"Predicted price for the new house: ${predicted_price:,.2f}") #Predicted Price: $510,000.00
