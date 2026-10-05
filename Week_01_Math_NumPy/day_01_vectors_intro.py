"""
Day 1: Vectors and NumPy Basics
Real-world use:
AI models convert text into lists of numbers (because AI don't understand text or documents) called vectors .
We need fast vector operations to calculate similarity between documents.
"""

import numpy as np

# Create dummy embedding vectors for two documents
doc_a = np.array([2.5, 3.0, 4.8])
doc_b = np.array([1.4, 3.7, 5.6])

# Inspect vector structure and data type
print("--- Vector Inspection ---")
print("Vector A:            ", doc_a)
print("Vector B:            ", doc_b)
print("Shape (Dimensions):  ", doc_a.shape)
print("Data type:           ", doc_a.dtype)

# Element-wise addition of two vectors
# Logic: [2.5 + 1.4, 3.0 + 3.7, 4.8 + 5.6]
print("\n--- Vector Operations ---")
combined_vector = doc_a + doc_b
print("Combined (A + B):    ", combined_vector)


"""Dot Product
  Dot product is the foundation of cosine similarity and semantic search"""
# Calculate dot product of two vectors
# Logic: (2.5 * 1.4) + (3.0 * 3.7) + (4.8 * 5.6) = 3.5 + 11.1 + 26.88 = 41.48
dot_result = np.dot(doc_a, doc_b)
print("Dot Product (A . B): ", dot_result)
print("-------------------------")
