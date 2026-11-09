"""
Day 22: Data Collection & Text Preprocessing
Project: Fake News Detection System
Loads the dataset (FR-01), cleans the text (FR-02)
and combines title + text (FR-03).
"""

import pandas as pd
import re
import string
import joblib

#  for fr-02: remove source tags so the model cannot cheat
def remove_source_leakage(text):
    text = re.sub(r'^.{0,80}\(reuters\)\s*-?\s*', '', text, flags=re.IGNORECASE)
    for word in ['reuters', 'featured image', 'getty images', 'via', 'image', 'video',
                 '21st century wire', 'twitter', 'pic']:
        text = re.sub(rf'\b{word}\b', '', text, flags=re.IGNORECASE)
    return text

# fr-02: lowercase, remove punctuation, numbers, extra spaces
def clean_text(text):
    text = remove_source_leakage(str(text))
    text = text.lower()
    text = re.sub(f'[{string.punctuation}]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# for fr-01: dataset statistics
def get_dataset_stats(df):
    return {
        "total_records": len(df),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_records": int(df.duplicated(subset=['text']).sum()),
        "class_distribution": {
            "Fake": int((df['label'] == 0).sum()),
            "Genuine": int((df['label'] == 1).sum()),
        },
    }
    
def load_and_preprocess_data():
    # fr-01: load both files
    print("\nstep 1: loading True.csv and Fake.csv...")
    try:
        df_true = pd.read_csv('True.csv')
        df_false = pd.read_csv('Fake.csv')
    except FileNotFoundError as e:
        print(f"ERROR: {e.filename} not found. Put True.csv and Fake.csv in the project folder.")
        raise

    # fr-01: labels (1 = genuine, 0 = fake), then combine and shuffle
    df_true['label'] = 1
    df_false['label'] = 0
    df = pd.concat([df_true, df_false], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # fr-01: show missing values, duplicates, class distribution (before cleaning)
    stats = get_dataset_stats(df)
    print("\n-~-~-~-Dataset Summary (FR-01)-~-~-~-")
    print(f"Total records     : {stats['total_records']}")
    print(f"Columns           : {', '.join(stats['columns'])}")
    print(f"Missing values    : {sum(stats['missing_values'].values())}")
    print(f"Duplicate records : {stats['duplicate_records']}")
    print(f"Fake news         : {stats['class_distribution']['Fake']}")
    print(f"Genuine news      : {stats['class_distribution']['Genuine']}")

    # fr-02: handle missing values and duplicates
    df = df.dropna(subset=['text'])
    df = df.drop_duplicates(subset=['text'])

    # fr-03: combine title and text
    df['text'] = df['title'].fillna('') + ' ' + df['text'].astype(str)

    # fr-02: clean the text
    print("\nstep 2: cleaning and preprocessing the text data...")
    df['clear_text'] = df['text'].apply(clean_text)

    # fr-04: remove articles that became identical after cleaning
    df = df.drop_duplicates(subset=['clear_text']).reset_index(drop=True)

    # fr-02: numbers after cleaning (used by the app)
    stats["records_after_cleaning"] = len(df)
    stats["class_distribution_after"] = {
        "Fake": int((df['label'] == 0).sum()),
        "Genuine": int((df['label'] == 1).sum()),
    }

    # fr-02: show summary after cleaning
    print("\n-~-~-~-After Cleaning (FR-02)-~-~-~-")
    print(f"Records left      : {stats['records_after_cleaning']}")
    print(f"Fake news         : {stats['class_distribution_after']['Fake']}")
    print(f"Genuine news      : {stats['class_distribution_after']['Genuine']}")

    # save stats so the app can show them without the csv files
    joblib.dump(stats, 'dataset_stats.pkl')

    print("Preprocessing completed successfully!")
    return df

if __name__ == "__main__":
    df = load_and_preprocess_data()
    print("\nSample of cleaned text:")
    print(df[['title', 'clear_text']].head(3).to_string(max_colwidth=60))