"""
Day 7: Custom PCA Engine (Dimensionality Reduction from Scratch)
Project: E-commerce Customer Behavior Compression (4D -> 2D)
"""

import numpy as np
np.set_printoptions(formatter={'float': lambda x: f"{x:.2f}"})

# 5 Customers dataset (Rows = Customers, Columns = 4 Features)
# Features: [Income (k$), Monthly Spend (k$), Orders/Month, Return Rate (%)]
X = np.array([
    [65.0, 11.0, 24.5, 31.0],
    [25.0, 13.0, 21.5, 37.0],
    [35.0, 14.0, 26.5, 34.0],
    [85.0, 18.0, 23.5, 30.0]
])


# 1. Dataset Inspection & Centering Data
print("┌" + "─" * 43 + "┐")
print("│  1. Dataset Inspection & Centering Data   │")
print("└" + "─" * 43 + "┘")

# Compute mean of each feature column (axis=0)
mean_vector = np.mean(X, axis=0)

# Center the data around the origin by subtracting mean
X_centered = X - mean_vector

print("Original 4D Customer Data:\n", X)
print("\nMean of Each Feature:\n", mean_vector)
print("\nCentered Data:\n", X_centered)
print("─" * 45)

# 2. Compute Covariance Matrix
print("\n┌" + "─" * 43 + "┐")
print("│  2. Compute Covariance Matrix             │")
print("└" + "─" * 43 + "┘")

# Number of samples 
n_samples = X.shape[0]

# Compute covariance matrix manually using matrix multiplication
cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)

print("Covariance Matrix (Shape: 4x4):\n", cov_matrix)
print("─" * 45)

# 3. Eigen-Decomposition & 2D Projection
print("\n┌" + "─" * 43 + "┐")
print("│  3. Eigen-Decomposition & 2D Projection   │")
print("└" + "─" * 43 + "┘")

# Compute eigenvalues and eigenvectors of covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
eigenvalues = eigenvalues.real
eigenvectors = eigenvectors.real

# Sort eigenvectors by eigenvalues in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

# Select top 2 eigenvectors 
projection_matrix = sorted_eigenvectors[:, :2]

# Project centered 4D data into 2D space (PC1, PC2)
X_pca_2d = X_centered @ projection_matrix

print("Sorted Eigenvalues (Variance per Component):\n", sorted_eigenvalues)
print("\nProjection Matrix W (Top 2 Eigenvectors, Shape 4x2):\n", projection_matrix)
print("\nFinal Compressed 2D Customer Data (Shape 4x2):\n", X_pca_2d)
print("─" * 45)

# Attach customer labels back to explain practical business view
customer_labels = ["Customer 1", "Customer 2", "Customer 3", "Customer 4"]
print("\nFinal 2D Customer Map (Ready for 2D Scatter Plot):")
for label, coords in zip(customer_labels, X_pca_2d):
    print(f"{label} -> PC1 (Spread): {coords[0]:6.2f}, PC2: {coords[1]:6.2f}")








