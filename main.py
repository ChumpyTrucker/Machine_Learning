import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Generate some sample data for demonstration purposes
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2 * X + 3 + np.random.randn(100, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the model's accuracy (in a real-world scenario, you would use appropriate evaluation metrics)
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# Plot the training data, test data, and the model's prediction line
plt.scatter(X_train, y_train, color='b', label="Training Data")
plt.scatter(X_test, y_test, color='g', label="Test Data")
plt.plot(X_test, y_pred, color='r', label="Model Prediction")
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()