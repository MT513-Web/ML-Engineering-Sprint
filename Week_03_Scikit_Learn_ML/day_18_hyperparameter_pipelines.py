"""
Day 18: Hyperparameter Tuning & ML Pipelines
Core Topics: Scikit-Learn Pipeline, StandardScaler, RandomForestClassifier, GridSearchCV
"""

import numpy as np                                                 # numerical operations and random data generation
import pandas as pd                                                # data manipulation and dataframe creation
from sklearn.model_selection import train_test_split, GridSearchCV # split & hyperparameter search
from sklearn.preprocessing import StandardScaler                   # feature scaling
from sklearn.ensemble import RandomForestClassifier                # ensemble model
from sklearn.pipeline import Pipeline                              # ML pipeline architecture
from sklearn.metrics import classification_report, accuracy_score  # evaluation metrics

# Set Pandas display options for clean terminal output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# ---- 1. Generate Synthetic Dataset ----
np.random.seed(42)                          
n_samples = 700

# Features (Input Variables X)
tenure_months = np.random.randint(1, 60, n_samples)
monthly_charges = np.random.uniform(25, 130, n_samples)
support_calls = np.random.randint(0, 9, n_samples)

# Target calculation logic (Binary outcome: 0 = Stayed, 1 = Churned)
churn_score = (0.03 * support_calls) - (0.06 * tenure_months) + (0.018 * monthly_charges)
churn_labels = (churn_score > 0.6).astype(int)

df = pd.DataFrame({
    'tenure_months': tenure_months,
    'monthly_charges': monthly_charges,
    'support_calls': support_calls,
    'churn': churn_labels
})

# ---- 2. Separate Features/Target & Train-Test Split ----
X = df[['tenure_months', 'monthly_charges', 'support_calls']]
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=45, stratify=y
)

# ---- 3. Build Scikit-Learn Pipeline ----
# Scaler + Classifier sequential steps
pipeline = Pipeline([
    ('scaler', StandardScaler()),                            # Feature scaling
    ('classifier', RandomForestClassifier(random_state=45))  # Random forest model
])

# ---- 4. Define Hyperparameter Search Grid ----
# Grid parameters for RandomForestClassifier inside the pipeline
param_grid = {
    'classifier__n_estimators': [50, 100, 150],     # number of decision trees in forest
    'classifier__max_depth': [3, 5, 8],             # maximum depth of trees
    'classifier__min_samples_split': [2, 5]         # min samples required to split internal node
}

# ---- 5. Execute GridSearchCV with 5-Fold Cross Validation ----
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv = 5,
    scoring= 'accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)                    # run exhaustive grid search over training folds

# ---- 6. Evaluate Best Model on Unseen Test Data ----
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\n" + "="*70)
print("  DAY 18: HYPERPARAMETER TUNING & PIPELINES EVALUATION")
print("="*70)
print(f"Best Hyperparameters Found : {grid_search.best_params_}")
print(f"Best 5-Fold CV Accuracy    : {grid_search.best_score_ * 100:.2f}%")
print(f"Test Set Accuracy Score    : {accuracy_score(y_test, y_pred) * 100:.2f}%\n")

print("Detailed Classification Report (Best Pipeline):")
print(classification_report(y_test, y_pred, target_names=['Stayed (0)', 'Churned (1)']))
print("="*70 + "\n")


