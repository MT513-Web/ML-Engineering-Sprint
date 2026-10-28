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

def load_and_preprocess_data(file_path):
    print("step 1: loading the dataset...")
    # read the dataset into a pandas dataframe
    df  = pd.read_csv(file_path)
    
    print(f"Original dataset shape: {df.shape}")
    
    # print the columns to check available column names in the CSV
    print("Dataset Columns:", df.columns)
    
    print(f"Original dataset shape: {df.shape}")
    
    # step 2: drop missing values to prevent errors during processing
    df = df.dropna(subset=['text'])
    
    # Ensure all text entries are strings (convert any numbers/floats if present)
    df['text'] = df['text'].astype(str)
    
    # step 3: define text cleaning function
    def clean_text(text):
        # convert text to lowercase
        text = text.lower()
        
        # rmv puncctuation marks
        text = re.sub(f'[{string.punctuation}]', '', text)
        
        # rmv numbers since they are usually not helpful in fake news detection
        text =  re.sub(r'\d+', '', text)
        
        # rmv extra white space
        text = re.sub(r'\s+', '', text).strip()
        
        return text
    
    print("step 2: cleaning and preprocessing the text data...")
    # apply the cleaning function to the text column
    df['clear_text'] = df['text'].apply(clean_text)
    
    print("Preprocessing completed successfully!")
    return df

if __name__ == "__main__":
    file_path = 'news.csv'
    clean_df = load_and_preprocess_data(file_path)
    print(clean_df.head())




 