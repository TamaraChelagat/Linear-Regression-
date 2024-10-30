import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = r'C:\Users\ADMIN\PycharmProjects\machineLearning\Nairobi Office Price Ex (1).csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("Data Preview:")
print(data.head())

# Extracting relevant columns
X = data['SIZE'].values
y = data['PRICE'].values

# Calculate and display the mean of office sizes and prices
mean_size = np.mean(X)
mean_price = np.mean(y)
print(f"\nMean of Office Size: {mean_size}")
print(f"Mean of Office Price: {mean_price}")

# Define Mean Squared Error function
def compute_mse(y_true, y_pred):
    """Calculate the Mean Squared Error between actual and predicted values."""
    return np.mean((y_true - y_pred) ** 2)

# Define Gradient Descent function
def perform_gradient_descent(X, y, m, c, learning_rate):
    """Perform one step of gradient descent and return updated parameters."""
    n = len(y)
    y_pred = m * X + c
    dm = (-2 / n) * np.sum(X * (y - y_pred))  # partial derivative w.r.t m
    dc = (-2 / n) * np.sum(y - y_pred)        # partial derivative w.r.t c
    m -= learning_rate * dm                   # update m
    c -= learning_rate * dc                   # update c
    return m, c

# Initialize parameters
m, c = np.random.rand(), np.random.rand()   # random initialization
learning_rate = 0.0001
epochs = 10

# Training loop
print("\nTraining the Model:")
for epoch in range(epochs):
    # Calculate predictions
    y_pred = m * X + c
    # Calculate Mean Squared Error
    mse = compute_mse(y, y_pred)
    print(f"Epoch {epoch + 1}: Mean Squared Error = {mse}")
    # Update weights using Gradient Descent
    m, c = perform_gradient_descent(X, y, m, c, learning_rate)

# Plotting the line of best fit
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, m * X + c, color='red', label='Line of Best Fit')
plt.xlabel('Size (sq. ft.)')
plt.ylabel('Price')
plt.legend()
plt.title('Linear Regression - Line of Best Fit after Final Epoch')
plt.show()

# Prediction for an office of size 100 sq. ft.
predicted_price_100_sq_ft = m * 100 + c
print(f"\nPredicted price for an office of size 100 sq. ft.: {predicted_price_100_sq_ft}")
