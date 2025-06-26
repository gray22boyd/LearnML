import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. Load and prepare dataset
print("\n📥 Loading California Housing Dataset...")
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target, name='Price')
print("First few rows of data:")
print(X.head())
print("Target variable preview:")
print(y.head())

# 2. Feature scaling (standardization)
print("\n⚖️ Scaling features (mean = 0, std = 1)...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Sample scaled feature (first row):", X_scaled[0])

# 3. Add bias term (column of ones)
print("\n➕ Adding bias (intercept) column to features...")
X_scaled = np.c_[np.ones(X_scaled.shape[0]), X_scaled]
print("Shape after adding bias:", X_scaled.shape)

# 4. Train-test split
print("\n🔀 Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print("Train set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])

# 5. Cost function
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X @ theta
    error = predictions - y
    cost = (1 / (2 * m)) * np.sum(error ** 2)
    return cost

# 6. Gradient Descent
def gradient_descent(X, y, theta, alpha=0.01, iterations=100):
    m = len(y)
    cost_history = []

    for i in range(iterations):
        predictions = X @ theta
        error = predictions - y
        gradients = (1 / m) * X.T @ error
        theta -= alpha * gradients
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

        if i % 10 == 0 or i == iterations - 1:
            print(f"Iteration {i:03}: Cost = {cost:.4f}")

    return theta, cost_history

# 7. Initialize weights (theta)
print("\n🔧 Initializing weights to zeros...")
theta = np.zeros(X_train.shape[1])
print("Initial theta:", theta)

# 8. Train model
print("\n🏋️ Starting Gradient Descent Training...")
theta, cost_history = gradient_descent(X_train, y_train, theta, alpha=0.1, iterations=100)

# 9. Final model weights and test error
print("\n✅ Training Complete.")
final_cost = compute_cost(X_test, y_test, theta)
print("Final test cost (MSE):", round(final_cost, 4))
print("\n📈 Final Model Weights:")
print(pd.Series(theta, index=["bias"] + list(housing.feature_names)))
