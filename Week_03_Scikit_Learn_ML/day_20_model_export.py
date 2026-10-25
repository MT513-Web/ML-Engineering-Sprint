"""
Day 20: Model Persistence & Exporting Trained Pipelines
Core Topics: Scikit-Learn Pipeline, RandomForestClassifier, Joblib Model Serialization (.joblib)
"""

import os                                               # operating system module for path management
import joblib                                           # model serialization library for saving python objects
import numpy as np                                      # numerical array operations
import pandas as pd                                     # dataframe creation and handling
from sklearn.model_selection import train_test_split    # dataset splitting utility
from sklearn.preprocessing import StandardScaler        # feature normalization scaler
from sklearn.ensemble import RandomForestClassifier     # ensemble classifier model
from sklearn.pipeline import Pipeline                   # sequential ml pipeline architecture

# Set Pandas display settings for clean terminal output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


# ---- 1. Generate Synthetic Customer Churn Dataset ----
np.random.seed(42)                                      
n_samples = 600                                         

# Input features
tenure_months = np.random.randint(1, 60, n_samples)         # customer tenure duration in months
monthly_charges = np.random.uniform(30, 140, n_samples)     # monthly bill amount in USD
support_calls = np.random.randint(0, 8, n_samples)          # customer support interaction count

# Target calculation logic (0 = Stayed, 1 = Churned)
churn_score = (0.35 * support_calls) - (0.05 * tenure_months) + (0.015 * monthly_charges)
churn_labels = (churn_score > 0.7).astype(int)

df = pd.DataFrame({
    'tenure_months':   tenure_months,
    'monthly_charges': monthly_charges,
    'support_calls':   support_calls,
    'churn':           churn_labels
})

# ---- 2. Split Features (X) and Target (y) ----
X = df[['tenure_months', 'monthly_charges', 'support_calls']]
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ---- 3. Build Scaler + Classifier Pipeline ----
# Combining preprocessing and model training into a single deployable artifact
churn_pipeline = Pipeline([
    ('scaler', StandardScaler()),                        # Step 1: Normalize feature distributions
    ('model', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)) # Step 2: Random Forest
])

# ---- 4. Train the Pipeline on Training Set ----
print("\n" + "="*70)
print("  DAY 20: TRAINING AND EXPORTING MODEL PIPELINE")
print("="*70)
churn_pipeline.fit(X_train, y_train)                     # fit scaler and model sequentially on training data
print("Training Completed Successfully!")

# ---- 5. Export/Serialize Trained Model to Disk ----
# Create a dedicated directory for saved models if it doesn't exist
model_dir = "Week_03_Scikit_Learn_ML/saved_models"
os.makedirs(model_dir, exist_ok=True)                    # create folder safely

model_filename = os.path.join(model_dir, "churn_pipeline.joblib")
joblib.dump(churn_pipeline, model_filename)              # serialize python pipeline object into binary file

print(f"Model successfully saved to disk at: {model_filename}")
print("="*70 + "\n")
