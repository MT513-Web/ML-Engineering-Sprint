"""
Day 23 & 24: Feature Extraction & Supervised Model Training
Project: Fake News Detection System
This script loads the preprocessed dataset, applies TF-IDF vectorization,
splits the data, and trains supervised classification models (Logistic Regression, Naive Bayes, SVM).
"""

import pandas as pd                                                # data handling and manipulation
from sklearn.model_selection import train_test_split               # splitting data into training and testing sets
from sklearn.feature_extraction.text import TfidfVectorizer        # converting text into numerical TF-IDF features
from sklearn.linear_model import LogisticRegression                # logistic Regression classification model
from sklearn.naive_bayes import MultinomialNB                      # naive Bayes classification model for text
from sklearn.svm import LinearSVC                                  # Support Vector Machine classifier for text
from sklearn.metrics import accuracy_score, classification_report  # evaluating model performance      
from day_22_preprocessing import load_and_preprocess_data          # custom function to load cleaned data
import joblib                                                      # saving and loading trained models

def train_and_evaluate_models():
    print("step 1: loading and processing data...")
    # load combined and cleaned dataframe from day 22 script
    df = load_and_preprocess_data()
    
    # define features (X) and target labels (y)
    X = df['clear_text']
    y = df['label']
    
    print("\nstep 2: splitting dataset into train and test sets...")
    # split data into 80% training and 20% testing sets using train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nstep 3: applying TF-IDE vectorization...")
    # dictionary of supervised models to train and evaluate
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print("\nstep 4: training supervised models...")
    # dictionary of supervised models to train and evaluate
    models = {
        "Logistic Regression": LogisticRegression(),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Support Vector Machine (SVM)": LinearSVC()
    }
    
    # loop through each model, train it, and print evaluation metrics
    for name, model in models.items():
        print(f"~~~ Training {name} ~~~")
        model.fit(X_train_tfidf, y_train)
        
        # make pridicitions on test data
        y_pred = model.predict(X_test_tfidf)
        
        # calculate and display accuracy and performance report
        acc = accuracy_score(y_test, y_pred) 
        print(f"Accuracy: {acc * 100:.2f}%")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # save the best performing model (SVM) and vectorizer for deployment
        joblib.dump(models["Support Vector Machine (SVM)"], 'fake_news_model.pkl') 
        joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
        print("\nModel and Vectorizer saved successfully using Joblib!")

if __name__ == "__main__":
    train_and_evaluate_models()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    