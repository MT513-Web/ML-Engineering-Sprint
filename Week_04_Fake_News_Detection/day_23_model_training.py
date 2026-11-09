"""
Day 23, 24, 25...: Feature Extraction, Model Training and Evaluation
Project: Fake News Detection System
Splits the data (FR-04), builds TF-IDF features (FR-03), trains and compares
3 models (FR-05), evaluates them (FR-07) and tests robustness (FR-08).
"""

import string
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)
from day_22_preprocessing import load_and_preprocess_data, clean_text

LABEL_NAMES = {0: "Fake", 1: "Genuine"}


# for fr-07: collect a few wrongly classified articles
def get_misclassified(test_df, y_pred, n=5):
    mask = test_df['label'].values != y_pred
    wrong = test_df[mask].head(n)
    wrong_pred = y_pred[mask][:n]
    examples = []
    for (_, row), pred in zip(wrong.iterrows(), wrong_pred):
        examples.append({
            "title": row['title'],
            "text": row['text'][:300],
            "actual": LABEL_NAMES[row['label']],
            "predicted": LABEL_NAMES[pred],
        })
    return examples


# for fr-08: change the text slightly and see if predictions change
def run_robustness_test(model, vectorizer, test_df, n=500):
    sample = test_df.sample(n=min(n, len(test_df)), random_state=42)
    variants = {
        "Original": lambda t: t,
        "UPPERCASE": lambda t: t.upper(),
        "Extra spaces": lambda t: t.replace(" ", "   "),
        "No punctuation": lambda t: t.translate(str.maketrans("", "", string.punctuation)),
    }
    results = {}
    original_pred = None
    for name, change in variants.items():
        texts = sample['text'].apply(change).apply(clean_text)
        pred = model.predict(vectorizer.transform(texts))
        if original_pred is None:
            original_pred = pred
        results[name] = {
            "accuracy": accuracy_score(sample['label'], pred),
            "unchanged": float((pred == original_pred).mean()),
        }
    return results


def train_and_evaluate_models():
    # fr-01, fr-02, fr-03: load, clean and combine title + text (day 22)
    print("step 1: loading and processing data...")
    df = load_and_preprocess_data()

    # fr-04: stratified 80/20 split before fitting TF-IDF (no data leakage)
    print("\nstep 3: splitting dataset into train and test sets...")
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )
    X_train, y_train = train_df['clear_text'], train_df['label']
    X_test, y_test = test_df['clear_text'], test_df['label']

    # fr-04: check that no article appears in both sets
    overlap = len(set(X_train) & set(X_test))
    print(f"Train: {len(train_df)} | Test: {len(test_df)} | Overlapping articles: {overlap}")

    # fr-03: TF-IDF, fitted on training data only
    print("\nstep 4: applying TF-IDF vectorization...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # fr-05: the three required models
    print("\nstep 5: training supervised models...")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Support Vector Machine (SVM)": LinearSVC(dual=False),
    }

    all_metrics = {}
    misclassified = {}

    for name, model in models.items():
        print(f"--- Training {name} ---")
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)

        # fr-07: accuracy, precision, recall, F1, confusion matrix
        all_metrics[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

        # fr-07: keep some wrong predictions for error analysis
        misclassified[name] = get_misclassified(test_df, y_pred)

    # fr-05: comparison table
    print("\n~-.~-.~-.~ Model Comparison (FR-05) ~-.~-.~-.~")
    table = pd.DataFrame(all_metrics).T[["accuracy", "precision", "recall", "f1"]]
    print((table * 100).round(2).to_string())

    # fr-05: best model is chosen by F1-score
    best_name = max(all_metrics, key=lambda m: all_metrics[m]["f1"])
    best_model = models[best_name]
    print(f"\nBest model (highest F1): {best_name}")

    # fr-07: confusion matrix of the best model
    cm = all_metrics[best_name]["confusion_matrix"]
    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(f"{'':<16}{'Pred Fake':>10}{'Pred Genuine':>14}")
    print(f"{'Actual Fake':<16}{cm[0][0]:>10}{cm[0][1]:>14}")
    print(f"{'Actual Genuine':<16}{cm[1][0]:>10}{cm[1][1]:>14}")

    # fr-08: robustness test on the best model
    print("\n~-.~-.~-.~ Robustness Test (FR-08) ~-.~-.~-.~")
    robustness = run_robustness_test(best_model, vectorizer, test_df)
    for variant, r in robustness.items():
        print(f"{variant:<15} accuracy: {r['accuracy']*100:.2f}% | "
              f"same prediction as original: {r['unchanged']*100:.2f}%")

    # save everything the app needs (no CSV needed after this)
    results = {
        "metrics": all_metrics,
        "best_model": best_name,
        "misclassified": misclassified,
        "robustness": robustness,
    }
    joblib.dump(results, 'evaluation_results.pkl')
    joblib.dump(best_model, 'fake_news_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    print("\nModel, vectorizer and results saved successfully using Joblib!")


if __name__ == "__main__":
    train_and_evaluate_models()