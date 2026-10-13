"""
Day 9: Production Data Cleaning & Missing Value Imputation
Core Topics: Missing Data Audit (MCAR/MAR), Median vs Mode Imputation, Text Sanitization
"""

import numpy as np
import pandas as pd

# Clean display configuration
pd.set_option("display.max_columns", 10)
pd.set_option("display.width", 1000)

print("\n┌" + "─" * 46 + "┐")
print("│  1. Dirty Dataset Generation & Missing Audit │")
print("└" + "─" * 46 + "┘")

# Reproducible synthetic raw data (1,000 customer records)
np.random.seed(35)
n_samples = 1000


# Generating raw data with intentional production defects:
#  Outliers & Skewness in Income
#  Random NaNs (Missing values)
#  Messy dirty strings with irregular whitesp
raw_income = np.random.exponential(scale=35000, size=n_samples) + 15000 
raw_income[
    np.random.choice(n_samples,size=50, replace=False)
] = np.nan # 5% missing
raw_income[0] = 2_5_000 # Extreme outlier (High skew)

raw_age = np.random.randint(18, 65, size=n_samples).astype(float)
raw_age[np.random.choice(n_samples, size=30, replace=False)]=np.nan

raw_cities = np.random.choice(
    ["  lahore ", "Karachi", "ISLAMABAD  ", "lahore", "karachi "],
    size=n_samples,
)
raw_cities = list(raw_cities)
for idx in np.random.choice(n_samples, size=40, replace=False):
    raw_cities[idx] = None # Missing categorical entries

df = pd.DataFrame(
    {
        "customer_id": np.arange(1001, 1001 + n_samples),
        "age": raw_age,
        "monthly_income": raw_income,
        "city": raw_cities,
    }
)

# Missing Data Audit Function
def audit_missing_data(data: pd.DataFrame) -> pd.DataFrame:
    """Calculates missing count and missing percentage per column."""
    missing_count = data.isnull().sum()
    missing_percent = (missing_count / len(data)) * 100
    audit_df = pd.DataFrame(
        {
            "Missing Count": missing_count,
            "Missing %": missing_percent.round(2),
            "Data Type": data.dtypes,
        }
    )
    return audit_df[audit_df["Missing Count"] > 0]

print("Missing Data Audit Report:")
print(audit_missing_data(df))
print("─" * 48)

# 2. Categorical Cleaning & Mode Imputation  ┘
print("\n┌" + "─" * 46 + "┐")
print("│  2. Categorical Cleaning & Mode Imputation   │")
print("└" + "─" * 46 + "┘")

# remove leading/trailing whitespace and normalize case
df["city"] = df["city"].str.strip().str.title()

# Compute Mode (Most Frequent City)
city_mode = df["city"].mode()[0]
print(f"Calculated City Mode: '{city_mode}'")

# Impute missing values with Mode
df["city"] = df["city"].fillna(city_mode)

# Optimize memory
df["city"] = df["city"].astype("category")

print("\nCleaned City Value Counts:")
print(df["city"].value_counts())
print(f"Missing values remaining in 'city': {df['city'].isnull().sum()}")
print("─" * 48)

# 3. Numeric Imputation: Mean vs Median Skew  
print("\n┌" + "─" * 46 + "┐")
print("│  3. Numeric Imputation: Mean vs Median Skew  │")
print("└" + "─" * 46 + "┘")

# Inspect Statistical Skew for Monthly Income
income_mean = df["monthly_income"].mean()
income_median = df["monthly_income"].median()

print(f"Income Arithmetic Mean:   PKR {income_mean:,.2f}")
print(f"Income 50th-Percentile:   PKR {income_median:,.2f}")
print(f"Distortion / Gap:         PKR {abs(income_mean - income_median):,.2f}")

# Impute Income using Median (Resistant to the 2.5M Outlier)

df["monthly_income"] = df["monthly_income"].fillna(income_median)

# Impute Age using Mean (Symmetric Distribution)
age_mean = df["age"].mean()
df["age"] = df["age"].fillna(age_mean).round(1)

# Final Data Cleanliness Verification
print("\nFinal Missing Data Audit:")
final_audit = audit_missing_data(df)
if final_audit.empty:
    print("Zero Missing Values Remaining! Dataset is 100% Clean.")
else:
    print(final_audit)

print("\nProcessed Clean Sample Records (First 5 Rows):")
print(df.head())
print("─" * 48)




