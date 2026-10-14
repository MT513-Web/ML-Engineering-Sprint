"""
Day 10: High-Performance Aggregations, GroupBy Pipelines & Multi-Level Indexing
Core Topics: Split-Apply-Combine, Multi-Metric .agg(), MultiIndex Flattening, Group Transforms
"""

import numpy as np
import pandas as pd

# display settings
pd.set_option("display.max_columns", 10)
pd.set_option("display.width", 1000)

# ----1. Synthetic Dataset & Single-Key GroupBy----
print("\n┌" + "─" * 46 + "┐")
print("│  1. Synthetic Dataset & Single-Key GroupBy   │")
print("└" + "─" * 46 + "┘")

# Reproducible transactional dataset (5,000 orders)
np.random.seed(42)
n_orders = 5000

cities = ["Lahore", "Karachi", "Islamabad", "Faisalabad", "Multan"]
segments = ["Budget", "Regular", "High-Value VIP"]

df = pd.DataFrame(
    {
    "Order_ID": np.arange(10001, 10001 + n_orders),
    "City": np.random.choice(cities, size=n_orders, p=[0.35, 0.30, 0.15, 0.10, 0.10]),
    "Customer_Segment": np.random.choice(segments, size=n_orders, p=[0.40, 0.45, 0.15]),
    "Order_Value": np.random.exponential(scale=1800, size=n_orders) + 200,
    "Items_Count": np.random.randint(1, 8, size=n_orders),
}
    )

# Downcasting for memory optimization
df["City"] = df["City"].astype("category")
df["Customer_Segment"] = df["Customer_Segment"].astype("category")
df["Items_Count"] = pd.to_numeric(df["Items_Count"], downcast="integer")
df["Order_Value"] = df["Order_Value"].round(2)

# Basic Single-Key GroupBy: Total Revenue per City
city_revenue = df.groupby("City", observed=False)["Order_Value"].sum().sort_values(ascending=False)

print("Total Revenue by City (PKR):")
print(city_revenue.apply(lambda x: f"PKR {x:,.2f}"))
print("─" * 48)


# ----2. Multi-Metric Aggregations (.agg())----       

print("\n┌" + "─" * 46 + "┐")
print("│  2. Multi-Metric Aggregations (.agg())       │")
print("└" + "─" * 46 + "┘")

# Multi-metric named aggregation across City and Customer Segment
city_segment_summary = (
    df.groupby(["City", "Customer_Segment"], observed=False)
    .agg(
        total_revenue=("Order_Value", "sum"),
        avg_order_value=("Order_Value", "mean"),
        spend_volatility_std=("Order_Value", "std"),
        total_orders=("Order_ID", "count"),
        avg_items_per_order=("Items_Count", "mean"),
    )
    .reset_index()  # Flatten the nested MultiIndex back to standard columns
)
city_segment_summary = city_segment_summary.sort_values(
    by="total_revenue", ascending=False
)

print("Aggregated City & Segment Business Metrics (Top 5 Records):")
print(city_segment_summary.head())
print("─" * 48)

# ----3. Group Transforms & Feature Engineering----
print("\n┌" + "─" * 46 + "┐")
print("│  3. Group Transforms & Feature Engineering   │")
print("└" + "─" * 46 + "┘")

# Calculate City-level average order value without collapsing rows
df["city_avg_order_value"] = df.groupby("City", observed=False)["Order_Value"].transform("mean").round(2)

# Engineer a new relative metric: Difference from City Average
df["diff_from_city_avg"] = (df["Order_Value"] - df["city_avg_order_value"]).round(2)

# Normalized Spending Ratio: (Order Value / City Mean)
df["spending_ratio"] = (df["Order_Value"] / df["city_avg_order_value"]).round(2)

print("Sample Feature-Engineered Transactions (First 5 Rows):")
print(
    df[
        [
            "Order_ID",
            "City",
            "Customer_Segment",
            "Order_Value",
            "city_avg_order_value",
            "diff_from_city_avg",
            "spending_ratio",
        ]
    ].head()
)
print("─" * 48)




