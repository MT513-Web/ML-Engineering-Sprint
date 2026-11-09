"""
Streamlit web app: Fake News Detection System
Tabs: Overview (FR-01, FR-09), Prediction (FR-06, FR-10),
Model Comparison (FR-05, FR-07, FR-09), Error Analysis (FR-07, FR-08).
"""

import joblib
import pandas as pd
import streamlit as st
from day_22_preprocessing import clean_text
from style import load_css, hero, stat_card, result_card
from charts import class_distribution_chart, model_comparison_chart, confusion_matrix_chart
from samples import SAMPLES

st.set_page_config(page_title="Fake News Detection System", page_icon="📰", layout="centered")
load_css()


# fr-10: load the saved files once; show a clear message if any is missing
@st.cache_resource
def load_assets():
    model = joblib.load('fake_news_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    results = joblib.load('evaluation_results.pkl')
    stats = joblib.load('dataset_stats.pkl')
    return model, vectorizer, results, stats


try:
    model, vectorizer, results, stats = load_assets()
except FileNotFoundError as e:
    st.error(f"File not found: {e.filename}. "
             "Please run day_22_preprocessing.py and day_23_model_training.py first.")
    st.stop()

metrics = results["metrics"]
best_name = results["best_model"]

hero("📰 Fake News Detection System",
     "Check whether a news article is potentially fake or potentially genuine.")

tab1, tab2, tab3, tab4 = st.tabs(
    ["🔍 Prediction", "📊 Overview", "🤖 Model Comparison", "🧪 Error Analysis"]
)

# tab 1: prediction (fr-06, fr-10)
# put chosen sample into the text box
def load_sample(text):
    st.session_state.news_text = text


with tab1:
    st.subheader("Check a News Article")
    news_input = st.text_area(
        "Paste the article text here:",
        placeholder="Paste at least 50 words for a reliable result...",
        height=200,
        key="news_text",
    )
    check_clicked = st.button("Check News")

    # sample articles
    st.markdown("#### Try a sample")
    cols = st.columns(len(SAMPLES))
    for col, sample in zip(cols, SAMPLES):
        with col:
            st.button(sample["label"], on_click=load_sample,
                      args=(sample["text"],), key=sample["label"],
                      width="stretch")
            st.caption(f"Expected: {sample['expected']}")

    if check_clicked:
        # fr-10: check the input first
        if news_input.strip() == "":
            st.warning("Please enter some news text before checking.")
        elif len(news_input.split()) < 50:
            st.warning("Please paste a full article (at least 50 words) for a reliable result.")
        else:
            # clean the text the same way as in training, then predict
            vec = vectorizer.transform([clean_text(news_input)])
            prediction = model.predict(vec)[0]

            # fr-06: show one of the two categories
            if prediction == 1:
                result_card("genuine", "✅ Potentially Genuine",
                            "The writing style is similar to genuine news articles.")
            else:
                result_card("fake", "⚠️ Potentially Fake",
                            "The writing style is similar to fake news articles.")

    #fr-06: this note must always be visible
    st.caption("This prediction is based on text patterns learned from the dataset. "
               "It does not independently verify the factual accuracy of the article.")

# tab 2: dataset overview (fr-01, fr-09) 
with tab2:
    st.subheader("Dataset Overview")

    # number cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        stat_card("Total records", f"{stats['total_records']:,}")
    with c2:
        stat_card("Missing values", sum(stats['missing_values'].values()))
    with c3:
        stat_card("Duplicates", f"{stats['duplicate_records']:,}")
    with c4:
        stat_card("Records after cleaning", f"{stats['records_after_cleaning']:,}")

    st.write("")

    # fake vs genuine graphs, before and after cleaning
    g1, g2 = st.columns(2)
    with g1:
        st.pyplot(class_distribution_chart(stats['class_distribution'], "Before cleaning"))
    with g2:
        st.pyplot(class_distribution_chart(stats['class_distribution_after'], "After cleaning"))

    with st.expander("Dataset columns"):
        st.write(", ".join(stats['columns']))

# tab 3: model comparison (fr-05, fr-07, fr-09) 
with tab3:
    st.subheader("Model Comparison")

    # fr-05: comparison table
    rows = []
    for name, m in metrics.items():
        rows.append({
            "Model": name,
            "Accuracy": f"{m['accuracy'] * 100:.2f}%",
            "Precision": f"{m['precision'] * 100:.2f}%",
            "Recall": f"{m['recall'] * 100:.2f}%",
            "F1-score": f"{m['f1'] * 100:.2f}%",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.success(f"Best model (highest F1-score): {best_name}")

    # fr-09: comparison graph
    st.pyplot(model_comparison_chart(metrics))

    # fr-07, fr-09: one confusion matrix per model
    st.markdown("#### Confusion Matrices")
    cols = st.columns(len(metrics))
    for col, (name, m) in zip(cols, metrics.items()):
        with col:
            st.pyplot(confusion_matrix_chart(m["confusion_matrix"], name.split(" (")[0]))

# tab 4: error analysis and robustness (fr-07, fr-08)
with tab4:
    # fr-07: wrongly classified articles
    st.subheader("Misclassified Articles")
    chosen = st.selectbox("Select a model:", list(results["misclassified"].keys()),
                          index=list(metrics.keys()).index(best_name))
    wrong = results["misclassified"][chosen]
    if not wrong:
        st.info("No misclassified examples saved for this model.")
    for item in wrong:
        title = item['title']
    with st.expander(title if len(title) <= 80 else title[:80] + "..."):
            st.write(f"**Actual:** {item['actual']}  |  **Predicted:** {item['predicted']}")
            st.write(item['text'] + "...")

    # fr-08: robustness test
    st.subheader(f"Robustness Test ({best_name})")
    rob_rows = []
    for variant, r in results["robustness"].items():
        rob_rows.append({
            "Text change": variant,
            "Accuracy": f"{r['accuracy'] * 100:.2f}%",
            "Same prediction as original": f"{r['unchanged'] * 100:.2f}%",
        })
    st.dataframe(pd.DataFrame(rob_rows), hide_index=True, width="stretch")
    st.caption("Capital letters, extra spaces and punctuation are normalised by the "
               "cleaning step, so predictions stay the same.")