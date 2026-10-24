"""
Day 19: Unsupervised Learning - K-Means Clustering & PCA Dimensionality Reduction
Core Topics: Synthetic Unlabeled E-Commerce Data, K-Means Clustering, Elbow Method, PCA (2D)
"""

import numpy as np                                      # numerical array calculations and random generation
import pandas as pd                                     # dataframe structures and data manipulation
from sklearn.preprocessing import StandardScaler        # standardizing features (Mean=0, Variance=1)
from sklearn.cluster import KMeans                      # K-Means clustering algorithm
from sklearn.decomposition import PCA                   # Principal Component Analysis for dimensionality reduction

# Set Pandas display options for readable output in terminal
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# ---- 1. Generate Synthetic Unlabeled Customer Behavioral Dataset ----
# Note: Unsupervised learning does NOT have any target 'y' label (e.g., no churn column)
np.random.seed(42)                                      # fixing random seed for reproducible outputs
n_samples = 600                                         # sample size 

# Creating multi-dimensional feature columns
annual_income_k = np.random.uniform(15, 120, n_samples)  # feature 1: customer annual income in $1000s
spending_score = np.random.uniform(1, 100, n_samples)    # feature 2: customer spending index (1 to 100)
purchase_frequency = np.random.randint(1, 50, n_samples) # feature 3: total orders placed per year

# Combine unlabelled features into a single Pandas DataFrame
df = pd.DataFrame({
    'annual_income_k': annual_income_k,                # input feature 1
    'spending_score': spending_score,                  # input feature 2
    'purchase_frequency': purchase_frequency           # input feature 3
})

print("\n" + "="*70)
print("  DAY 19: UNSUPERVISED LEARNING - UNLABELED CUSTOMER DATASET")
print("="*70)
print(df.head())                                        # print top 5 rows of input data
print("-" * 70 + "\n")

# ---- 2. Feature Scaling (Critical step for Distance-Based Models) ----
# K-Means calculates Euclidean distance; features with large scales would dominate the clusters
scaler = StandardScaler()                              # initialize standard scaler instance
X_scaled = scaler.fit_transform(df)                    # scale features to standard normal distribution (Z-score)

# ---- 3. K-Means Clustering Implementation ----
# We group customers into 3 behavioral segments (K = 3)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10) # initialize K-Means with 3 clusters
df['Cluster_Label'] = kmeans.fit_predict(X_scaled)        # fit model on scaled features & assign cluster IDs (0, 1, 2)

print("  K-MEANS CLUSTERING SUMMARY (K = 3 Clusters)")
print("="*70)
print("Customer Count per Discovered Cluster:")
print(df['Cluster_Label'].value_counts())                 # count of customers assigned to each cluster
print("-" * 70)

# Calculate average feature values per cluster to understand business personas
cluster_profile = df.groupby('Cluster_Label').mean().round(2)
print("Discovered Customer Behavioral Personas (Mean Values):")
print(cluster_profile)                                 # cluster centers profiling table
print("-" * 70 + "\n")

# ---- 4. PCA (Principal Component Analysis) Dimensionality Reduction ----
# Compress 3 features (income, spending, frequency) into 2 Principal Components (PC1 & PC2)
pca = PCA(n_components=2, random_state=42)                # initialize PCA to reduce dimensions to 2
X_pca = pca.fit_transform(X_scaled)                       # compute principal components transformation

# Store PCA outputs into dataframe for easy inspection
df['PC1'] = X_pca[:, 0]                                # captures maximum variance
df['PC2'] = X_pca[:, 1]                                # captures second most variance

print("  PCA DIMENSIONALITY REDUCTION SUMMARY")
print("="*70)
print(f"Explained Variance Ratio by PC1 & PC2: {np.round(pca.explained_variance_ratio_ * 100, 2)}%")
print(f"Total Preserved Variance Score     : {pca.explained_variance_ratio_.sum() * 100:.2f}%")
print("-" * 70)
print("Sample Compressed 2D Representation (Top 5 rows):")
print(df[['annual_income_k', 'spending_score', 'purchase_frequency', 'PC1', 'PC2', 'Cluster_Label']].head())
print("="*70 + "\n")











