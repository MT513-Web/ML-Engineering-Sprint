# Machine Learning Engineering Sprint

Welcome to my machine learning and data engineering sprint repository! This project chronicles my journey through core ML mathematics, advanced data pipelines, supervised/unsupervised learning, and deploying production-grade web applications.

## Sprint Overview

* **Core Focus:** Linear Algebra, Pandas, Scikit-Learn Pipelines, NLP Classification, and Streamlit/FastAPI Deployments.
* **Architecture:** Modular codebase organized by weekly milestones and real-world capstone projects.
* **Standards:** Clean documentation, reproducible scripts, and robust data cleaning.

---

## Capstone Projects Showcase

Here are the three main end-to-end projects built during this sprint:

### 1. E-Commerce Production EDA & Analytics Dashboard (Week 2)
* **What it does:** An interactive Streamlit app for exploratory data analysis, statistical distribution checking, Tukey's fence outlier detection, and real-time executive KPI reporting on e-commerce transaction data.
* **Live Demo:** [E-Commerce Analytics Hub](https://ecommerceanalyticsdashboard-2dqe3dbtdwxveq5h4bugdx.streamlit.app)

### 2. Customer Churn Prediction Engine (Week 3)
* **What it does:** A complete ML solution predicting customer churn using Scikit-Learn pipelines, backed by a FastAPI endpoint and Docker containerization. It features an interactive Streamlit UI for single evaluations and bulk CSV batch uploads.
* **Live Demo:** [Customer Churn Prediction Engine](https://customer-churn-web-app-debn7cekcrda8mwprjtova.streamlit.app/)

### 3. Fake News Detection System (Week 4)
* **What it does:** An SRS-compliant NLP text classification system that cleans news datasets (removing source leakage tags like "Reuters" to prevent cheating), applies TF-IDF vectorization, trains and evaluates three models (Logistic Regression, Naive Bayes, and SVM), and features a multi-tab Streamlit dashboard with confusion matrices and error analysis.
* **Live Demo:** *Coming soon / Add link later*

---

## Sprint Roadmap & Tracker

| Week | Day | Focus Topic | Key Deliverables | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Week 1** | Day 01 | Vectors & Dot Product | Vector representations, element-wise math, and dot product | Done |
| | Day 02 | Matrices & Transformations | Matrix operations, dot products, and feature transformations | Done |
| | Day 03 | Vector Norms & Distance | Euclidean distance, Manhattan distance, and cosine similarity | Done |
| | Day 04 | Matrix Inversion & Solvers | Determinants, inverse matrices, and systems of linear equations | Done |
| | Day 05 | Eigenvalues & Eigenvectors | Conceptual foundations for dimensionality reduction (PCA) | Done |
| | Day 06 | Probability Foundations | Bayes Theorem, conditional probability, and distributions | Done |
| | Day 07 | Project: Math Engine | Comprehensive math and vector processing library | Done |
| **Week 2** | Day 08 | Production Pandas & Vectorization | Vectorized operations, memory profiling, and high-performance filtering | Done |
| | Day 09 | Data Cleaning & Missing Imputation | Handling MCAR/MAR patterns, robust imputation, and format sanitization | Done |
| | Day 10 | Outlier Detection & Robust Scaling | IQR, Z-score thresholds, RobustScaler, and distribution normalization | Done |
| | Day 11 | Feature Engineering & Encodings | One-Hot, frequency encoding, cyclical datetime signals, and feature crosses | Done |
| | Day 12 | GroupBy Aggregations & Analytics | Pivot tables, rolling window metrics, multi-index aggregations, and joins | Done |
| | Day 13 | Automated Data Pipeline Class | Modular OOP pipeline architecture, validation, error handling, and logging | Done |
| | Day 14 | Project: Streamlit Analytics Dashboard | Interactive customer intelligence dashboard with cached data exploration | Done |
| **Week 3** | Day 15 | ML Workflow & Regressors | Supervised vs Unsupervised, Train/Test Split, Linear & Ridge Regression | Done |
| | Day 16 | Classification Algorithms | Logistic Regression, Decision Trees, Random Forest & Evaluation Metrics | Done |
| | Day 17 | Model Evaluation & Cross-Validation | K-Fold CV, Precision, Recall, F1-Score, ROC-AUC Curves | Done |
| | Day 18 | Hyperparameter Tuning & Pipelines | GridSearchCV, RandomizedSearchCV, Scikit-Learn Pipeline & ColumnTransformer | Done |
| | Day 19 | Unsupervised Learning | K-Means Clustering, PCA Dimensionality Reduction | Done |
| | Day 20 | Model Persistence & API Serving | Model Export (Joblib/Pickle), FastAPI Endpoint for Inference | Done |
| | Day 21 | End-to-End ML Production Project | Predictive ML Web Service with Streamlit & Model Pipeline | Done |
| **Week 4** | Day 22 | Data Collection & Text Preprocessing | News dataset loading, cleaning, source leakage removal, missing values and duplicate handling | Done |
| | Day 23 | Feature Extraction & Model Training | TF-IDF Vectorization, Logistic Regression, Naive Bayes, and Support Vector Machine (LinearSVC) training | Done |
| | Day 24 | Model Evaluation & Persistence | Accuracy, Precision, Recall, F1-score, Confusion Matrix, and model/metrics persistence via Joblib | Done |
| | Day 25 | Streamlit Web App & Deployment | Multi-tab interactive UI (Prediction, Overview, Comparison, Error Analysis) and live deployment | Done |

---

## Tech Stack & Environment

* **Language:** Python 3.10+
* **Core Libraries:** NumPy, Pandas, Scikit-Learn, Streamlit, FastAPI, Joblib
* **Tools:** VS Code, Git, GitHub, Docker