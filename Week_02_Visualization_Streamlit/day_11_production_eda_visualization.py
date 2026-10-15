"""
Day 11: Production Exploratory Data Analysis (EDA) & Data Visualization
Core Topics: Distribution & Skewness (KDE), Outliers & IQR (Boxplots), Correlation Heatmaps
Libraries: Matplotlib, Seaborn, Pandas, NumPy
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Visual styling for clean production figures
sns.set_theme(style='whitegrid', palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

# ---- 1. Synthetic Dataset & Distribution Check ----
print("\n┌" + "─" * 46 + "┐")
print("│  1. Synthetic Dataset & Distribution Check   │")
print("└" + "─" * 46 + "┘")

# Realistic dataset (2,000 customers)
np.random.seed(42)
n_records = 2000

# Continuous features
age = np.random.normal(loc=35, scale=10, size=n_records).clip(18, 70).round(1)
items_purchased = np.random.randint(1, 10, size=n_records)

# Right-skewed spending (Exponential base + direct correlation with items purchased)
spending_amount = (
    np.random.exponential(scale=1500, size=n_records)
    + (items_purchased * 350)
    + 500
).round(2)

# Categorical feature
membership_tier = np.random.choice(
    ["Bronze", "Silver", "Gold"], size=n_records, p=[0.50, 0.35, 0.15]
)

df = pd.DataFrame(
    {
        "customer_id": np.arange(2001, 2001 + n_records),
        "age": age,
        "items_purchased": items_purchased,
        "spending_amount": spending_amount,
        "membership_tier": membership_tier,
    }
)

df["membership_tier"] = df["membership_tier"].astype("category")

# Calculate Skewness metrics
spending_skew = df["spending_amount"].skew()
age_skew = df["age"].skew()

print(f"Spending Amount Skewness: {spending_skew:.2f} (Right-Skewed / Positive)")
print(f"Age Feature Skewness:     {age_skew:.2f} (Near Symmetric / Bell Curve)")

# Plotting Dimension 1: Histogram + KDE Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot A: Age (Symmetric Distribution)
sns.histplot(df["age"], kde=True, ax=axes[0], color="#2b5c8f", bins=25)
axes[0].set_title(
    f"Age Distribution (Skew: {age_skew:.2f})", fontsize=12, fontweight="bold"
)
axes[0].set_xlabel("Customer Age")
axes[0].set_ylabel("Frequency Count")

# Plot B: Spending Amount (Right-Skewed Distribution)
sns.histplot(
    df["spending_amount"], kde=True, ax=axes[1], color="#d95f02", bins=30
)
axes[1].set_title(
    f"Spending Distribution (Skew: {spending_skew:.2f})",
    fontsize=12,
    fontweight="bold",
)
axes[1].set_xlabel("Spending Amount (PKR)")
axes[1].set_ylabel("Frequency Count")

plt.tight_layout()
plt.savefig("eda_step1_distribution.png", dpi=300)
print("\nDistribution chart successfully saved as 'eda_step1_distribution.png'")
print("─" * 48)
plt.show()

# ----2. Outlier Detection & Boxplot Analysis ----     
print("\n┌" + "─" * 46 + "┐")
print("│  2. Outlier Detection & Boxplot Analysis     │")
print("└" + "─" * 46 + "┘")

# Mathematical IQR Outlier Detection on Spending Amount
q1 = df["spending_amount"].quantile(0.25)
q3 = df["spending_amount"].quantile(0.75)
iqr = q3 - q1

lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)

outliers = df[(df["spending_amount"] < lower_bound) | (df["spending_amount"] > upper_bound)]
outlier_count = len(outliers)
outlier_pct = (outlier_count / len(df)) * 100

print(f"25th Percentile (Q1):       PKR {q1:,.2f}")
print(f"75th Percentile (Q3):       PKR {q3:,.2f}")
print(f"Interquartile Range (IQR):  PKR {iqr:,.2f}")
print(f"Lower Boundary (Min Floor): PKR {lower_bound:,.2f}")
print(f"Upper Boundary (Outlier Cutoff): PKR {upper_bound:,.2f}")
print(f"Identified Outliers:        {outlier_count} records ({outlier_pct:.2f}%)")

# Plotting Dimension 2: Boxplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot B: Segmented Boxplot by Membership Tier
sns.boxplot(
    x="membership_tier",
    y="spending_amount",
    data=df,
    ax=axes[1],
    palette="Set2"
)

axes[1].set_title("Spending Spread by Membership Tier", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Membership Tier")
axes[1].set_ylabel("Spending Amount (PKR)")

plt.tight_layout()
plt.savefig("eda_step2_boxplots.png", dpi=300)
print("\nBoxplot chart successfully saved as 'eda_step2_boxplots.png'")
print("─" * 48)
plt.show()

# ----3. Correlation & Relationship Analysis----      
print("\n┌" + "─" * 46 + "┐")
print("│  3. Correlation & Relationship Analysis      │")
print("└" + "─" * 46 + "┘")

# Pearson Correlation Matrix for numeric features
numeric_cols = ["age", "items_purchased", "spending_amount"]
corr_matrix = df[numeric_cols].corr()

print("Pearson Correlation Matrix:")
print(corr_matrix.round(3))

# Plotting Dimension 3: Relationships
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot A: Scatter Plot with Linear Regression Fit
sns.regplot(
    data=df,
    x="items_purchased",
    y="spending_amount",
    ax=axes[0],
    scatter_kws={"alpha": 0.3, "color": "#1b9e77"},
    line_kws={"color": "#d95f02", "linewidth": 2},
)
axes[0].set_title(
    "Items Purchased vs Spending Amount (Regression)",
    fontsize=12,
    fontweight="bold",
)
axes[0].set_xlabel("Items Purchased (Count)")
axes[0].set_ylabel("Spending Amount (PKR)")

# Plot B: Annotated Correlation Heatmap
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    center=0,
    square=True,
    linewidths=0.5,
    ax=axes[1],
    cbar_kws={"shrink": 0.8},
)
axes[1].set_title(
    "Numeric Feature Correlation Heatmap", fontsize=12, fontweight="bold"
)

plt.tight_layout()
plt.savefig("eda_step3_correlation.png", dpi=300)
print("\nCorrelation charts successfully saved as 'eda_step3_correlation.png'")
print("─" * 48)
plt.show()