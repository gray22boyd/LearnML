import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
import os

# Add parent folder to path to import model
sys.path.append(os.path.join(os.path.dirname(__file__), 'linear_regression'))
from linear_regression.housing_linear_regression import compute_cost, gradient_descent

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target, name='Price')

# Use only 1 feature for 2D & 3D clarity (e.g., MedInc)
X_feature = X[['MedInc']].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_feature)
X_scaled = np.c_[np.ones(X_scaled.shape[0]), X_scaled]  # add bias term

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
theta_init = np.zeros(X_train.shape[1])
theta, _ = gradient_descent(X_train, y_train, theta_init, alpha=0.1, iterations=100)

# User input for visualization mode
mode = input("Which plot would you like to see? Enter '2d', '3d', or 'both': ").strip().lower()

if mode in ('2d', 'both'):
    # ----------- 2D Plot: Predictions vs Actual -----------
    X_full = np.c_[np.ones(X_scaled.shape[0]), X_scaled[:, 1]]
    y_pred = X_full @ theta

    fig1 = px.scatter(
        x=X_feature['MedInc'], y=y,
        labels={'x': 'Median Income', 'y': 'House Price'},
        title="Predicted vs Actual Prices"
    )
    fig1.add_scatter(x=X_feature['MedInc'], y=y_pred, mode='lines', name='Prediction')
    fig1.show()

if mode in ('3d', 'both'):
    # ----------- 3D Plot: Cost Surface over w and b -----------
    w_vals = np.linspace(-3, 3, 50)
    b_vals = np.linspace(-3, 3, 50)
    W, B = np.meshgrid(w_vals, b_vals)
    Z = np.zeros_like(W)

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            t = np.array([B[i, j], W[i, j]])  # [bias, weight]
            Z[i, j] = compute_cost(X_scaled, y, t)

    fig2 = go.Figure(data=[
        go.Surface(z=Z, x=W, y=B, colorscale='Viridis', contours=dict(z=dict(show=True)))
    ])
    fig2.update_layout(
        title='Cost Function J(w, b)',
        scene=dict(
            xaxis_title='w (weight)',
            yaxis_title='b (bias)',
            zaxis_title='J(w, b)',
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.5)  # flatten the Z dimension
        ),
        scene_camera=dict(
            eye=dict(x=1.5, y=1.5, z=0.7)  # nice diagonal angle
        )
    )

    fig2.show()
