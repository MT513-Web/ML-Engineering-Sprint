""" 
Day 2: Matrices and Matrix Operations
Real-world use:
In Machine Learning, a matrix represents a complete dataset or feature table:
- Rows represent individual samples or documents.
- Columns represent features, dimensions, or word counts.
"""

import numpy as np

# Create a 2D Matrix (3 samples, 2 features each)
# Example: 3 students, [Study Hours, Test Score]
data_matrix = np.array ([
    [25.0, 76.9],
    [23.9, 87.4],
    [35.5, 26.1]
])

print(".~~~~ Matrix Inspection ~~~~.")
print("Matrix: \n", data_matrix)
print("Shapes (Rows, Cols): ", data_matrix.shape)
print("Dimensions (ndim):   ", data_matrix.ndim)
print("Data Type:           ", data_matrix.dtype)
print("Total Elements:      ", data_matrix.size)
print(".~~~~~~~~~~~~~~~~~~~~~~~~~~~.")

# --- Matrix Transpose & Slicing ---
print("\n.~~~ Transpose & Slicing ~~~.")

# Transpose: Swap rows and columns
transposed_matrix = data_matrix.T
print("Transposed Matrix (A^T):\n", transposed_matrix)
print("Original Shape:         ", data_matrix.shape)
print("Transposed Shape:       ", transposed_matrix.shape)

# Row extraction (First sample / student)
first_sample = data_matrix[0,:]
print("\nFirst Sample (Row 0):     ", first_sample)

# Column extraction (First feature across all samples: Study Hours)
first_feature = data_matrix[:, 0]
print("Feature 1 (All rows, COl 0): ", first_feature)
print(".~~~~~~~~~~~~~~~~~~~~~~~~~~~~.")


# --- Matrix Multiplication (Linear Transformation) ---
print("\n--- Matrix Multiplication ---")

# Feature Weights: 2 features converted into 1 single performance score
weights = np.array([
    [0.3],
    [0.7]
])

# Multiply: (3, 2) @ (2, 1) -> Resulting Shape: (3, 1)
# Each student's final score = (Hours * 0.3) + (Score * 0.7)
scores = data_matrix @ weights

print("Weights Matrix (Shape 2, 1):\n", weights)
print("\nCalculated Scores (Shape 3, 1):\n", scores)
print("Resulting Shape: ", scores.shape)
print("---------------------------------")