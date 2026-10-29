"""
Day 23: TF-IDF Feature Extraction & K-Means Clustering
This script loads the cleaned dataset, converts text data into 
numerical features using TF-IDF Vectorization, and groups the 
news articles using K-Means Clustering (Unsupervised Learning).
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from day_22_preprocessing import load_and_preprocess_data

def train_models():
    print("step 1: loading and processing data...")
    # load the cleaned dataframe from the preprocessing script
    df = load_and_preprocess_data('news.csv')
    
    # define text features (X)
    X = df['clear_text']
    
    print("step 2: applying TF-IDF vectorization...")
    # convert text documents to matrix of TF-IDF features
    vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = vectorizer.fit_transform(X)
    
    print("\nstep 3: Applying K-Means Clustering...")
    # group news into 2 clusters (unsupervised learning)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(X_tfidf)
    
    # add cluster results to dataframe
    df['cluster'] = kmeans.labels_
    print("Clustering completed successfully!")
    print(df[['clear_text', 'cluster']].head())

if __name__ == "__main__":
    train_models()