"""
Day 8: Production Pandas & High-Performance Vectorization
Core Topics: Memory Optimization, Vectorized Processing vs Loops, Boolean Masking
"""

import time
import numpy as np
import pandas as pd

# Set clean console display
pd.set_option("display.max_columns", 10)
pd.set_option("display.width", 1000)

# Generate a synthetic e-commerce transaction dataset (100,000 rows)
np.random.seed(42)
n_rows = 100_00


# 1. Memory Profiling & Downcasting          
print("\n┌" + "─" * 46 + "┐")
print("│  1. Memory Profiling & Downcasting           │")
print("└" + "─" * 46 + "┘")

df = pd.DataFrame(
    {
        "order_id": np.arange(1, n_rows + 1),
        "quantity": np.random.randint(1, 10, size=n_rows),
        "unit_price": np.random.uniform(10.0, 500.0, size=n_rows),
        "payment_method": np.random.choice(
            ["Credit Card", "EasyPaisa", "JazzCash", "COD"], size=n_rows
        ),
        "customer_rating": np.random.choice([1, 2, 3, 4, 5], size=n_rows),
    }
)

# Measure baseline memory usage
initial_memory_kb = df.memory_usage(deep=True).sum()/1024
print(f"Initial Memory Usage: {initial_memory_kb:.2f} KB")

# Optimize Memory:
# 1. Convert repetitive string objects to categorical type
df["payment_method"] = df["payment_method"].astype("category")

# 2. Downcast numeric columns to smaller byte sizes
df["quantity"] = pd.to_numeric(df["quantity"], downcast="integer")
df["customer_rating"] = pd.to_numeric(df["customer_rating"], downcast="integer")
df["unit_price"] = pd.to_numeric(df["unit_price"], downcast="float")

# Measure optimized memory usage
optimized_memory_kb = df.memory_usage(deep=True).sum() / 1024
memory_saved_percent = (
    (initial_memory_kb - optimized_memory_kb) / initial_memory_kb
) * 100

print(f"Optimized Memory Usage: {optimized_memory_kb:.2f} KB")
print(f"Memory Reduced By: {memory_saved_percent:.2f}%")
print("─" * 48)


# 2. Benchmarking: Python Loop vs Vectorized
print("\n┌" + "─" * 46 + "┐")
print("│  2. Loop vs Vectorization Benchmarking       │")
print("└" + "─" * 46 + "┘")

# Method 1: Traditional Python Loop (iterrows - Slow Anti-Pattern)
start_time = time.time()
loop_results = []
for _, row in df.iterrows():
    loop_results.append(row["quantity"] * row["unit_price"])
df["total_bill_loop"] = loop_results
loop_duration = time.time() - start_time

# Method 2: Pandas Vectorized Operation (Fast C-Level Parallel Execution)
start_time = time.time()
df["total_bill_vectorized"] = df["quantity"] * df["unit_price"]
vectorized_duration = time.time() - start_time

speedup = loop_duration / vectorized_duration

print(f"Loop Time (iterrows):   {loop_duration:.4f} seconds")
print(f"Vectorized Time:        {vectorized_duration:.4f} seconds")
print(f"Vectorization Speedup:  {speedup:.1f}x Faster!")
print("─" * 48)



# 3. Vectorized Conditional Logic (np.select) 
print("\n┌" + "─" * 46 + "┐")
print("│  3. Vectorized Business Logic (np.select)    │")
print("└" + "─" * 46 + "┘")

# Define business conditions (Boolean Masks)
conditions = [
    (df["total_bill_vectorized"] > 1500) & (df["customer_rating"] >= 4),
    (df["total_bill_vectorized"] > 800),
]

# Choices matching each condition
choices = ["High-Value VIP", "Regular Customer"]

# Default value if no condition is met
default_choice = "Budget Customer"

# Apply vectorized selection in a single operation
df["customer_segment"] = np.select(conditions, choices, default=default_choice)
df["customer_segment"] = df["customer_segment"].astype("category")

print("Customer Segment Distribution:\n")
print(df["customer_segment"].value_counts())
print("\nSample Processed Records (First 5 Rows):")
print(
    df[
        [
            "order_id",
            "quantity",
            "unit_price",
            "total_bill_vectorized",
            "customer_segment",
        ]
    ].head()
)

print("─" * 48)

