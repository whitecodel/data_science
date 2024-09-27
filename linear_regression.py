import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Example data
data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Scores': [50, 55, 60, 62, 70, 75, 78, 80, 85, 88]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Features (independent variable)
X = df[['Hours_Studied']]  # Use double brackets to keep X as a DataFrame

# Target (dependent variable)
y = df['Scores']

# Split the data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Model parameters
print(f"Slope (Coefficient): {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")

# Predict using the test set
y_pred = model.predict(X_test)

# Display predicted values
print(f"Actual Scores: {y_test.values}")
print(f"Predicted Scores: {y_pred}")

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Calculate R-squared (R²)
r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2}")

# Plotting the training data
plt.scatter(X_train, y_train, color='blue', label='Training Data')

# Plotting the test data
plt.scatter(X_test, y_test, color='green', label='Test Data')

# Plotting the regression line
plt.plot(X_train, model.predict(X_train), color='red', label='Regression Line')

# Adding labels and title
plt.xlabel('Hours Studied')
plt.ylabel('Scores')
plt.title('Linear Regression Model')
plt.legend()

# Display the plot
plt.show()

# Predict the score for a student who studied 7.5 hours
new_data = [[7.5]]
predicted_score = model.predict(new_data)
print(f"Predicted Score for 7.5 hours of study: {predicted_score[0]}")