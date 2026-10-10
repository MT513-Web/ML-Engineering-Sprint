"""
Day 6: Matrix Decomposition (Singular Value Decomposition - SVD)
Core Math: A = U @ Sigma @ Vt
Real ML Use: Recommender Systems (Collaborative Filtering) & Image Compression.
"""

import numpy as np

# Clean terminal formatting
np.set_printoptions(formatter={'float': lambda x: f"{x:.2f}"})

# User-Movie Rating Matrix: 3 Users (Rows) x 2 Movies (Columns)
# E.g., [Action Movie Rating, Sci-Fi Movie Rating]
A = np.array([
    [2.0, 5.3],
    [3.4, 1.7],
    [2.7, 6.8]
])

# 1. Perform Singular Value Decomposition
print("┌" + "─" * 43 + "┐")
print("│  1. Singular Value Decomposition (SVD)    │")
print("└" + "─" * 43 + "┘")

# np.linalg.svd returns:
# U = Eigenvectors of (A @ A.T) -> User latent space
# Vt = Eigenvectors of (A.T @ A) -> Movie latent space
# S = Square roots of non-zero Eigenvalues -> Concept strength
U, S, Vt = np.linalg.svd(A, full_matrices=False)

print("Original Data Matrix A:\n", A)
print("\nU Matrix (User Latent Features):\n", U)
print("\nSingular Values S (Concept Strengths):\n", S)
print("\nVt Matrix (Movie Latent Features):\n", Vt)
print("─" * 45)

# 2. Reconstruction: A = U @ diag(S) @ Vt
print("\n┌" + "─" * 43 + "┐")
print("│  2. Matrix Reconstruction & Check         │")
print("└" + "─" * 43 + "┘")

# Convert 1D singular values array S into a 2x2 Diagonal Matrix Sigma
Sigma = np.diag(S)

# Reconstruct original matrix
A_reconstructed = U @ Sigma @ Vt

print("Original Ratings Matrix A:\n", A)
print("\nReconstructed Matrix (U @ Sigma @ Vt):\n", A_reconstructed)

# Check if reconstruction matches original
is_match = np.allclose(A, A_reconstructed)
print("\nReconstruction Match: ", is_match)
print("─" * 45)

