# Fake News Detection System

A Streamlit web app that classifies a news article as **potentially fake** or
**potentially genuine** using TF-IDF features and machine-learning models.

**Live demo:** _add link after deployment_

> The app predicts from writing patterns learned from the dataset. It does not
> verify the factual accuracy of an article.

## Dataset

- **Name:** Fake and Real News Dataset
- **Source:** [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- **Files:** `True.csv` (genuine) and `Fake.csv` (fake)
- **Downloaded:** 5 November 2026
- Labels are used as provided by Kaggle and were not independently verified.
- The CSV files are not included in this repository because of their size.

## Pipeline

1. **Load and explore:** records, missing values, duplicates, class distribution
2. **Preprocess:** remove duplicates, lowercase, remove punctuation, numbers and
   extra spaces, remove stop words, remove source tags (such as "Reuters") that
   would let the model cheat
3. **Features:** TF-IDF on title + text (5,000 features), fitted on training data only
4. **Split:** stratified 80/20 train/test split, with a check that no article
   appears in both sets
5. **Models:** Logistic Regression, Multinomial Naive Bayes, Support Vector Machine
6. **Evaluation:** accuracy, precision, recall, F1-score, confusion matrix,
   misclassified examples, robustness test

## Results (test set: 7,729 articles)

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Logistic Regression | 97.85% | 97.53% | 98.58% | 98.05% |
| Multinomial Naive Bayes | 92.04% | 93.18% | 92.24% | 92.71% |
| **Support Vector Machine** | **98.36%** | **98.29%** | **98.73%** | **98.51%** |

The best model by F1-score is the SVM (LinearSVC). Precision and recall are for
the Genuine class. The robustness test uses a random sample of 500 test articles.

## App Tabs

| Tab | What it shows |
|---|---|
| Prediction | Paste an article, get Potentially Fake / Genuine, try sample articles |
| Overview | Dataset numbers and fake/genuine distribution graphs |
| Model Comparison | Comparison table, best model, graph, confusion matrices |
| Error Analysis | Misclassified articles and robustness test |

The sample articles in the Prediction tab are illustrative examples, not taken
from the dataset.

## Requirements Coverage

| FR | Where |
|---|---|
| FR-01 Dataset loading and exploration | `day_22_preprocessing.py`, Overview tab |
| FR-02 Preprocessing | `day_22_preprocessing.py` |
| FR-03 Feature extraction | `day_23_model_training.py` |
| FR-04 Dataset splitting | `day_23_model_training.py` |
| FR-05 Model training and comparison | `day_23_model_training.py`, Model Comparison tab |
| FR-06 News classification | Prediction tab |
| FR-07 Model evaluation | Model Comparison and Error Analysis tabs |
| FR-08 Robustness | Error Analysis tab |
| FR-09 Visualization | Overview and Model Comparison tabs |
| FR-10 UI and error handling | `app.py` |

## Run Locally

```bash
pip install -r requirements.txt

# only if you want to retrain (needs True.csv and Fake.csv in the folder)
python day_23_model_training.py

streamlit run app.py
```

## Project Structure

```
app.py                      Streamlit app (tabs and logic)
style.py                    Colours and CSS
charts.py                   Graphs
samples.py                  Sample articles
day_22_preprocessing.py     Loading and cleaning
day_23_model_training.py    Training and evaluation
.streamlit/config.toml      Theme
*.pkl                       Saved model, vectorizer and results
```

## Limitations

- The dataset is mostly US political news from 2016-2017, so results on other
  topics, sources, languages or short headlines may be weaker.
- High scores on this dataset do not mean the model is a reliable fake news
  detector in general.
- The model learns writing style, not facts.

## Credits

Virtual University of Pakistan, Information Retrieval and Machine Learning project.
Supervisor: Umra Naeem.