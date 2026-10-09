"""
Day 5: Eigenvalues & Eigenvectors
Core Math: A @ v = lambda * v
Real ML Use: Principal Component Analysis (PCA) & Dimensionality Reduction.
"""

import numpy as np
np.set_printoptions(formatter={'float': lambda x: f"{x:.1f}"})

# Define a 2x2 symmetric transformation matrix (e.g., covariance-like matrix)
A = np.array([
    [4.0, 2.0],
    [1.0, 3.0]
])

# 1. Compute Eigenvalues & Eigenvectors
print("┌" + "─" * 43 + "┐")
print("│  1. Compute Eigenvalues & Eigenvectors    │")
print("└" + "─" * 43 + "┘")

# np.linalg.eig() returns two things:
# 1. eigenvalues: array of scalar scaling factors
# 2. eigenvectors: matrix where each column is an eigenvector
eigenvalues, eigenvectors = np.linalg.eig(A)
eigenvalues = eigenvalues.real
eigenvectors = eigenvectors.real

print("Original Matrix A:\n", A)
print("\nEigenvalues (Scaling Factors):\n", eigenvalues)
print("\nEigenvectors (Direction Vectors as Columns):\n", eigenvectors)
print("─" * 45)


# 2. Mathematical Proof: A @ v == lambda
print("\n┌" + "─" * 43 + "┐")
print("│  2. Mathematical Proof: A @ v == lambda*v │")
print("└" + "─" * 43 + "┘")

# Clean imaginary parts if any
eigenvalues = eigenvalues.real
eigenvectors = eigenvectors.real

# Select first eigenvalue and its matching eigenvector (Column 0)
lambda_1 = eigenvalues[0]
v_1 = eigenvectors[:, 0]

# Left Hand Side: Matrix multiplication with eigenvector
lhs = A @ v_1

# Right Hand Side: Scalar multiplication with eigenvalue
rhs = lambda_1 * v_1

print("Selected Eigenvalue (lambda_1):", lambda_1)
print("Matching Eigenvector (v_1):     ", v_1)
print("\nLHS (A @ v_1):                  ", lhs)
print("RHS (lambda_1 * v_1):           ", rhs)

# Numerical check
is_verified = np.allclose(lhs, rhs)
print("\nProof Verification Match:       ", is_verified)

if is_verified:
    print("Verdict: Direction preserved! Vector only scaled by lambda.")
else:
    print("Verdict: Discrepancy found.")
print("─" * 45)
