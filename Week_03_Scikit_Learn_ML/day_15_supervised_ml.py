"""
Day 15: Supervised Machine Learning - Core Workflow & Linear Regression
Core Topics: Synthetic Data, Features/Target Split, Train-Test Split, Linear Model Training
"""

import pandas as pd                                      # numerical operations and random data generation
import numpy as np                                       # data manipulation and dataframe creation
from sklearn.model_selection import train_test_split     # splitting data into train and test sets
from sklearn.linear_model import LinearRegression        # linear regression algorithm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score     # evaluation metrics

# Force Pandas to display all columns in terminal
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# ---- 1. Generate Synthetic E-Commerce Dataset ----
np.random.seed(35)                                       # fixed seed for consistent results
n_samples = 500                                          # dataset size (500 rows)      

# Features (Input Variables X)
ad_spend = np.random.uniform(500, 5000, n_samples)
website_visits = np.random.randint(100, 2000, n_samples)
discount_offered = np.random.uniform(5, 30, n_samples)

# Target (Output Variable y) with linear formula + noise
noise = np.random.normal(0, 500, n_samples)               # random noise for real-world variation
sales_revenue = (2.5 * ad_spend) + (15 * website_visits) + (50 * discount_offered) + 1000 + noise

# Combine into Pandas DataFrame
df =  pd.DataFrame({
        'ad_spend': ad_spend,
        'website_visits': website_visits,
        'discount_offered': discount_offered,
        'sales_revenue': sales_revenue
})

# ---- 2. Display Sample Dataset Output ----
print("\n" + "="*50)
print("  DAY 15: SUPERVISED ML - SYNTHETIC DATASET SAMPLE")
print("="*50)
print(df.head())                           
print("-" * 50 + "\n")

# ---- 3. Separate Features (X) and Target (y) ----
X = df[['ad_spend', 'website_visits', 'discount_offered']]
y = df['sales_revenue']

# ---- 4. Perform Train-Test Split (80% Train, 20% Test) ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=35
)

print("  DATASET SPLIT SUMMARY")
print("="*70)
print(f"Total Rows In Dataset   : {len(df)}")
print(f"Training Features Shape : {X_train.shape}  (80% for model learning)")
print(f"Testing Features Shape  : {X_test.shape}   (20% for model evaluation)")
print("-" * 70 + "\n")

# ---- 6. Train Linear Regression Model ----
model =  LinearRegression()
model.fit(X_train, y_train)                      # learning patterns from training data

# ---- 7. Generate Predictions on Unseen Test Data ----
y_pred = model.predict(X_test)                   # predicting sales for test features

# ---- 8. Evaluate Model Performance Metrics ----
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("  MODEL PERFORMANCE EVALUATION")
print("="*70)
print(f"Mean Absolute Error (MAE)   : PKR {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): PKR {rmse:.2f}")
print(f"R-squared Score (R² Score)   : {r2:.4f} ({r2 * 100:.2f}% variance explained)")
print("-" * 70)

# Display Learned Weights (Coefficients)
print("\n  MODEL PARAMETERS (WEIGHTS & BIAS)")
print("="*70)
for col, coef in zip(X.columns, model.coef_):
    print(f"Weight for '{col}': {coef:.2f}")
print(f"Intercept (Bias)            : {model.intercept_:.2f}")
print("-" * 70 + "\n")













