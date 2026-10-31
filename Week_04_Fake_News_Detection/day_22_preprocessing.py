"""
Day 22: Data Collection & Text Preprocessing
Project: Fake News Detection System
This script loads the raw news dataset, handles missing values, 
and cleans the text data (lowercasing, removing punctuation, numbers, and extra spaces) 
to prepare it for TF-IDF vectorization.
"""

import pandas as pd
import re
import string

def load_and_preprocess_data():
    print("\nstep 1: loading True.csv and Fake.csv...")
   # Load both files from Kaggle dataset
    df_true = pd.read_csv('True.csv')
    df_false = pd.read_csv('Fake.csv')
    
    # Assign labels: 1 for True/Real news, 0 for Fake news
    df_true['label'] = 1
    df_false['label'] = 0
    
    # Combine both dataframes into one
    df = pd.concat([df_true, df_false], ignore_index=True)
    
    # Shuffle the dataset so fake and real articles mix up
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Combined dataset shape: {df.shape}")
    print("Dataset Columns:", df.columns)
    
    # drop missing values
    df = df.dropna(subset=['text'])
    
    # Ensure all text entries are strings
    df['text'] = df['text'].astype(str)
    
    # step 3: define text cleaning function
    def clean_text(text):
        text = text.lower()
        text = re.sub(f'[{string.punctuation}]', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    print("\nstep 2: cleaning and preprocessing the text data...")
    df['clear_text'] = df['text'].apply(clean_text)
    
    print("Preprocessing completed successfully!")
    return df

if __name__ == "__main__":
    df = load_and_preprocess_data()
    print(df.head())
    
     