"""
Day 7: PCA Engine from Scratch
Core Math: Standardization -> Covariance -> eigh() -> Explained Variance -> Projection
Use Case: Dimensionality Reduction for E-commerce Customer Segmentation
"""

import numpy as np
np.set_printoptions(formatter={'float': lambda x: f"{x:.2f}"})

# 1. Multi-Feature Customer Dataset (10 Customers x 4 Features)
# Features: [Income ($k), Monthly Spend ($k), Orders/Month, Return Rate (%)]
X = np.array([
    [65.0, 1.2, 5.0, 2.0],
    [70.0, 1.5, 6.0, 1.5],
    [30.0, 0.4, 2.0, 4.0],
    [32.0, 0.5, 2.0, 3.5],
    [85.0, 2.1, 8.0, 1.0],
    [28.0, 0.3, 1.0, 5.0],
    [90.0, 2.4, 9.0, 0.8],
    [45.0, 0.8, 3.0, 2.5],
    [75.0, 1.8, 7.0, 1.2],
    [38.0, 0.6, 2.0, 3.0]
])

print("┌" + "─" * 46 + "┐")
print("│  1. Feature Standardization (Z-Score)        │")
print("└" + "─" * 46 + "┘")

# Mean and Standard Deviation scaling
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X_scaled = (X - mean) / std

print("Original Features Shape:", X.shape)
print("Standardized Mean (approx 0):", np.mean(X_scaled, axis=0))
print("Standardized Std (all 1s):   ", np.std(X_scaled, axis=0))


# 2. Covariance Matrix & Symmetric Eig-Decomp 
print("\n┌" + "─" * 46 + "┐")
print("│  2. Covariance & Symmetric Eig-Decomp        │")
print("└" + "─" * 46 + "┘")

n_samples = X.shape[0]
cov_matrix = (X_scaled.T @ X_scaled) / (n_samples - 1)

# np.linalg.eigh is optimized for symmetric covariance matrices
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# eigh returns in ascending order, so reverse to descending
eigenvalues = eigenvalues[::-1]
eigenvectors = eigenvectors[:, ::-1]

# Calculate Explained Variance Ratio
total_variance = np.sum(eigenvalues)
explained_variance_ratio = (eigenvalues / total_variance) * 100
variance_2d = np.sum(explained_variance_ratio[:2])

print("Eigenvalues (Variance per Component):\n", eigenvalues)
print("\nExplained Variance Ratio (%):\n", explained_variance_ratio)
print(f"\nTotal Information Captured by PC1 + PC2: {variance_2d:.2f}%")


# 3. 2D Projection & Latent Space Output    
print("\n┌" + "─" * 46 + "┐")
print("│  3. 2D Projection Matrix & Transformed Data  │")
print("└" + "─" * 46 + "┘")

# Select Top 2 Eigenvectors (W Matrix: 4x2)
projection_matrix = eigenvectors[:, :2]

# Project Scaled Data to 2D
X_pca_2d = X_scaled @ projection_matrix

print("Projection Matrix W:\n", projection_matrix)
print("\nCompressed 2D Coordinates:\n", X_pca_2d)
print("─" * 48)