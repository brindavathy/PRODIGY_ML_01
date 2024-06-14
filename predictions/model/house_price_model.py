import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the training data
data = pd.read_csv("data/train.csv")

# Step 2: Find and print missing values
print("Missing values before handling:")
print(data.isnull().sum())

# Step 3: Drop rows with missing values in important columns
data = data.dropna(subset=["LotFrontage", "LotArea", "BedroomAbvGr", "FullBath", "YearBuilt", "SalePrice"])

# Step 4: Check and print missing values after handling
print("Missing values after handling:")
print(data.isnull().sum())

# Step 5: Select numeric data
numeric_data = data.select_dtypes(include=[np.number])

# Step 6: Display a heatmap of the correlation matrix
correlation_matrix = numeric_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

# Step 7: Define features and target for the linear regression model
X = data[['LotArea', 'BedroomAbvGr', 'FullBath', 'YearBuilt']]
y = data['SalePrice']

# Step 8: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 9: Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 10: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 11: Calculate and print evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R-squared (R2): {r2}')

# Calculate and print evaluation metrics on the training set for comparison
y_train_pred = model.predict(X_train)
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train, y_train_pred)

print(f'Training Mean Squared Error (MSE): {train_mse}')
print(f'Training Root Mean Squared Error (RMSE): {train_rmse}')
print(f'Training R-squared (R2): {train_r2}')

# Step 12: Load the test data
test_data = pd.read_csv("data/test.csv")

# Ensure the test data contains the required features and handle missing values if necessary
required_features = ['LotArea', 'BedroomAbvGr', 'FullBath', 'YearBuilt']
missing_cols = [col for col in required_features if col not in test_data.columns]

if missing_cols:
    raise ValueError(f"Missing columns in test data: {missing_cols}")

# Select the same features as the training data
test_data = test_data[required_features]

# Handle any missing values in the test data (e.g., drop rows with missing values)
test_data = test_data.dropna()

# Step 13: Predict prices for the new test data
predicted_prices = model.predict(test_data)

# Step 14: Print the predicted prices
print(predicted_prices)

# Ensure the directory exists
os.makedirs('predictions/model', exist_ok=True)

# Then save the model
with open('predictions/model/house_price_model.pkl', 'wb') as file:
    pickle.dump(model, file)
# Save the model to a file

print("Model saved to house_price_model.pkl")
