"""
Day 16: Supervised Classification - Logistic Regression & Decision Trees
Core Topics: Synthetic Classification Data, Features/Target Split, Train-Test Split, Model Training
"""

import numpy as np                                     # numerical operations and random data generation
import pandas as pd                                    # data manipulation and dataframe creation
from sklearn.model_selection import train_test_split   # splitting dataset into train and test sets
from sklearn.linear_model import LogisticRegression    # logistic regression classification algorithm
from sklearn.tree import DecisionTreeClassifier        # decision tree classification algorithm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # metrics

# Set Pandas display options for clean terminal output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# ---- 1. Generate Synthetic Customer Churn Dataset ----
np.random.seed(42)                          
n_samples = 600

# Features (Input Variables X)
tenure_months = np.random.randint(1, 48, n_samples)        # customer relationship duration in months
monthly_charges = np.random.uniform(20, 120, n_samples)    # monthly subscription in USD
support_calls = np.random.randint(0, 10, n_samples)        # number of customer support calls made

# Target (Binary Output Variable y: 0 = Stayed, 1 = Churned)
# Logical Rule: High support calls + short tenure + high charges increases churn probability
churn_prob = (0.35 * support_calls) - (0.08 * tenure_months) + (0.02 * monthly_charges)
churn_labels = (churn_prob > 0.5).astype(int)              # convert probability threshold to 0 or 1

# Combine into Pandas DataFrame
df = pd.DataFrame({
    'tenure_months': tenure_months,         
    'monthly_charges': monthly_charges,     
    'support_calls': support_calls,         
    'churn': churn_labels                   
})

# ---- 2. Display Sample Dataset & Target Distribution ----
print("\n" + "="*70)
print("  DAY 16: SUPERVISED CLASSIFICATION - CUSTOMER CHURN DATASET")
print("="*70)
print(df.head())                            # print top 5 rows
print("-" * 70)
print("Target Class Distribution (0 = Stayed, 1 = Churned):")
print(df['churn'].value_counts())           # print count of 0s and 1s
print("-" * 70 + "\n")

# ---- 3. Separate Features (X) and Target (y) ----
X = df[['tenure_months', 'monthly_charges', 'support_calls']]   # input feature columns (2D DataFrame)
y = df['churn']                                                 # target label column (1D Series)

# ---- 4. Perform Train-Test Split (80% Train, 20% Test) ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y           # stratify keeps class proportions equal
)

# ---- 5. Display Split Shapes and Class Balance ----
print("  DATASET SPLIT SUMMARY")
print("="*70)
print(f"Total Customer Records : {len(df)}")
print(f"Training Features Shape: {X_train.shape}  (80% for learning)")
print(f"Testing Features Shape : {X_test.shape}   (20% for evaluation)")
print("-" * 70)
print("Training Target Class Balance (0s vs 1s):")
print(y_train.value_counts(normalize=True).round(2))
print("-" * 70 + "\n")

# ---- 6. Train Logistic Regression Model ----
log_model = LogisticRegression()                  # logistic regression instance
log_model.fit(X_train, y_train)                   # training on 80% train data
y_pred_log = log_model.predict(X_test)            # predicting churn on test features

# ---- 7. Train Decision Tree Classifier ----
tree_model = DecisionTreeClassifier(max_depth=3, random_state=42) # decision tree instance
tree_model.fit(X_train, y_train)                  # training on 80% train data
y_pred_tree = tree_model.predict(X_test)          # predicting churn on test features

# ---- 8. Evaluate Basic Model Accuracy ----
acc_log = accuracy_score(y_test, y_pred_log)      # logistic regression accuracy
acc_tree = accuracy_score(y_test, y_pred_tree)    # decision tree accuracy

print("  MODEL ACCURACY COMPARISON")
print("="*70)
print(f"Logistic Regression Accuracy : {acc_log * 100:.2f}%")
print(f"Decision Tree Accuracy       : {acc_tree * 100:.2f}%")
print("-" * 70 + "\n")


