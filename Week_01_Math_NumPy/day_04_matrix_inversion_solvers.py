"""
Day 4: Matrix Inversion & Solvers (Linear Equations)
Real-world use:
- Finding weights (W) in Linear Regression without gradient descent: A @ W = B -> W = inv(A) @ B
- Determinant: Checks if the feature matrix has unique solutions (det != 0).
"""

import numpy as np
np.set_printoptions(formatter={'float': lambda x: f"{x:.1f}"})

# Feature Matrix A: 2 samples, 2 features (e.g., [Study Hours, Past Score])
# Shape: (2, 2)
A = np.array([
    [2.0, 1.0],
    [1.0, 3.0]
])

# Known Target vector B: Final Exam Marks for both students
# Shape: (2, 1)
B = np.array([
    [8.0],
    [14.0]
])


#  1. Matrix Inspection & Determinant     
print("┌" + "─" * 43 + "┐")
print("│  1. Matrix Inspection & Determinant       │")
print("└" + "─" * 43 + "┘")

# Determinant Formula for 2x2: (ad - bc)
# (2*3) - (1*1) = 6 - 1 = 5
det_A = np.linalg.det(A)

print("Matrix A (Features):\n", A)
print("\nVector B (Known Targets):\n", B)
print("\nDeterminant of A: ", round(det_A, 2))

if det_A != 0:
    print("Verdict: Determinant is Non-Zero. Matrix is Invertible!")
else:
    print("Verdict: Singular Matrix! Inversion not possible.")
print("─" * 45)

# 2. Matrix Inversion & Solving Weights
print("\n┌" + "─" * 43 + "┐")
print("│  2. Matrix Inversion & Solving Weights    │")
print("└" + "─" * 43 + "┘")

# Method 1: Explicit Inversion using np.linalg.inv()
A_inv = np.linalg.inv(A)
weights_inv = A_inv @ B

# Method 2: Recommended Numerical Solver (Faster & Numerically Stable)
weights_solve = np.linalg.solve(A, B) 

print("Inverse of Matrix A (A^-1):\n", A_inv)
print("\nCalculated Weights (via Inverse @ B):\n", weights_inv)
print("\nCalculated Weights (via np.linalg.solve):\n", weights_solve)

print("\nInterpretation:")
print(f" Weight 1 (Study Hours): {weights_solve[0][0]:.1f}")
print(f" Weight 2 (Past Score):  {weights_solve[1][0]:.1f}")
print("─" * 45)

# 3. Model Verification (Reconstruction)
print("\n┌" + "─" * 43 + "┐")
print("│  3. Model Verification (Reconstruction)   │")
print("└" + "─" * 43 + "┘")

# Verify: Does A @ weights equal B?
predicted_b = A @ weights_solve

print("Original Target B:\n", B)
print("\nReconstructed Target (A @ W):\n", predicted_b)

# Check if both match numerically
is_matching = np.allclose(predicted_b, B)
print("\nVerification Match: ", is_matching)

if is_matching:
    print("Verdict: Solved weights are 100% mathematically accurate!")
else:
    print("Verdict: Discrepancy found.")
print("─" * 45)





