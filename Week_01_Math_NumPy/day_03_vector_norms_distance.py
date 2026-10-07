"""
Day 3: Vector Norms, Distance, and Similarity
Real-world use:
- L2 Norm: Measures the magnitude (length) of an embedding vector.
- Euclidean Distance: Straight-line distance between two data points.
- Cosine Similarity: Directional similarity used in RAG & Vector Search.
"""

import numpy as np

# Create two feature vectors (e.g., embeddings of two search queries)
# Vector A: [Word Count, Keyword Frequency, Sentiment Score]
vec_a = np.array([3.0, 4.0])
vec_b = np.array([9.0, 6.0])

# 1.L2 Norm
print("┌" + "─" * 40 + "┐")
print("│  1. Vector Norms (Magnitude)           │")
print("└" + "─" * 40 + "┘")
# L2 Norm Formula: sqrt(x^2 + y^2)
norm_a = np.linalg.norm(vec_a)
norm_b = np.linalg.norm(vec_b)

print("Vector A:         ", vec_a)
print("Magnitude of A:   ", norm_a)  # sqrt(3^2 + 4^2) = 5.0
print("Vector B:         ", vec_b)
print("Magnitude of B:   ", norm_b)  # sqrt(6^2 + 8^2) = 10.0
print("─" * 40)


# 2. Euclidean Distance
print("\n┌" + "─" * 40 + "┐")
print("│  2. Euclidean Distance                 │")
print("└" + "─" * 40 + "┘")

# Straight-line distance between vec_a and vec_b
# Formula: sqrt((x2 - x1)^2 + (y2 - y1)^2) = norm(a - b)
euclidean_dist = np.linalg.norm(vec_a - vec_b)

print("Distance between A and B: ", euclidean_dist)
print("Interpretation: Less distance = closer points in feature space")
print("─" * 42)


# 3. Cosine Similarity (Semantic Search)
print("\n┌" + "─" * 40 + "┐")
print("│  3. Cosine Similarity (RAG Metric)     │")
print("└" + "─" * 40 + "┘")

# Formula: dot(a, b) / (norm(a) * norm(b))
dot_product = np.dot(vec_a,vec_b)
cosine_sim = dot_product / (norm_a * norm_b)

print("Dot Product (A . B):      ", dot_product)
print("Cosine Similarity:        ", cosine_sim)

if cosine_sim > 0.8:
    print("Verdict: Highly Similar / Relevant (Semantic Match)")
else:
    print("Verdict: Not closely related")
print("─" * 42)