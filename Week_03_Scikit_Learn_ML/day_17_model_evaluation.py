"""
Day 17: Model Evaluation & Cross-Validation
Core Topics: Confusion Matrix, Precision, Recall, F1-Score, Classification Report, K-Fold CV
"""

import numpy as np                                                                     # numerical array operations
import pandas as pd                                                                    # data manipulation
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold # CV tools
from sklearn.linear_model import LogisticRegression                                    # classification algorithm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report    # metrics

# Set Pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# ---- 1. Generate Realistic Imbalanced Customer Churn Data ----
np.random.seed(42)
n_samples = 800

# Features
tenure_months = np.random.randint(1, 60, n_samples)
monthly_charges = np.random.uniform(30, 150, n_samples)
support_calls = np.random.randint(0, 8, n_samples)

# Target logic with moderate noise (Realistic scenario)
churn_score = (0.28*support_calls) - (0.05*tenure_months) + (0.015*monthly_charges)
churn_labels = (churn_score > 0.8).astype(int)

df = pd.DataFrame({
    'tenure_months':   tenure_months,
    'monthly_charges': monthly_charges,
    'support_calls':   support_calls,
    'churn':           churn_labels
})

X = df[['tenure_months', 'monthly_charges', 'support_calls']]
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ---- 3. Model Training ----
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ---- 4. Basic Accuracy vs Advanced Classification Metrics ----
print("\n" + "="*70)
print("  DAY 17: MODEL EVALUATION & ADVANCED METRICS REPORT")
print("="*70)
print(f"Overall Test Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")

print("Confusion Matrix (TN, FP / FN, TP):")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("-" * 70)

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Stayed (0)', 'Churned (1)']))
print("="*70 + "\n")

# ---- 5. Perform 5-Fold Cross-Validation ----
strat_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=strat_kfold, scoring='accuracy')

print("  5-FOLD CROSS-VALIDATION PERFORMANCE")
print("="*70)
print(f"Scores for each of the 5 folds : {np.round(cv_scores * 100, 2)}")
print(f"Mean CV Accuracy Score        : {cv_scores.mean() * 100:.2f}%")
print(f"Standard Deviation Score      : (+/- {cv_scores.std() * 100:.2f}%)")
print("="*70 + "\n")

