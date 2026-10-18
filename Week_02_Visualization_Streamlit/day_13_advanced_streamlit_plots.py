"""
Day 13: Advanced Streamlit Dashboards & Embeddings (Matplotlib / Seaborn Integration)
Core Topics: Embedded Statistical EDA Plots, Dynamic Chart Rendering, st.pyplot
Execution Command: streamlit run day_13_advanced_streamlit_plots.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="Advanced Analytics & EDA Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Visual theme setup for Matplotlib/Seaborn
sns.set_theme(style="whitegrid", palette="muted")

# 2. Title & Context
st.title(" Embedded Statistical EDA Dashboard")
st.markdown(
    "Embedding static **Matplotlib/Seaborn** statistical visualizations into an interactive Streamlit application."
)
st.divider()


# 3. Cached Data Generator
@st.cache_data
def load_advanced_data():
    np.random.seed(42)
    n = 1200
    cities = ["Lahore", "Karachi", "Islamabad", "Faisalabad"]

    data = pd.DataFrame(
        {
            "customer_id": np.arange(5001, 5001 + n),
            "city": np.random.choice(cities, size=n, p=[0.35, 0.30, 0.20, 0.15]),
            "age": np.random.normal(38, 11, size=n).clip(18, 70).round(1),
            "items_purchased": np.random.randint(1, 10, size=n),
            "spending_amount": (
                np.random.exponential(scale=2000, size=n) + 500
            ).round(2),
        }
    )
    return data


df = load_advanced_data()

# 4. Sidebar Controls
st.sidebar.header(" Dashboard Controls")

selected_city = st.sidebar.selectbox(
    "Filter by City:", options=["All Cities"] + sorted(df["city"].unique().tolist())
)

plot_metric = st.sidebar.radio(
    "Select Metric for Distribution:", options=["spending_amount", "age"]
)

# Data filtering
filtered_df = df.copy()
if selected_city != "All Cities":
    filtered_df = filtered_df[filtered_df["city"] == selected_city]

# 5. KPI Overview Cards
col1, col2, col3 = st.columns(3)
col1.metric("Active Sample Size", f"{len(filtered_df):,} Customers")
col2.metric("Mean Spending", f"PKR {filtered_df['spending_amount'].mean():,.2f}")
col3.metric("Median Spending", f"PKR {filtered_df['spending_amount'].median():,.2f}")

st.divider()

# 6. Embedded Matplotlib / Seaborn Visualizations
st.subheader(" Statistical Visualizations")

tab1, tab2 = st.tabs(["Distributions & Spread", "Feature Relationships"])

with tab1:
    col_left, col_right = st.columns(2)

    # Chart A: Histogram + KDE
    with col_left:
        st.markdown(f"### Distribution of `{plot_metric}`")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(filtered_df[plot_metric], kde=True, ax=ax, color="#2b5c8f")
        ax.set_title(f"{plot_metric.title()} Distribution", fontweight="bold")
        st.pyplot(fig)  # Embed Matplotlib Figure in Streamlit

    # Chart B: Outlier Boxplot
    with col_right:
        st.markdown(f"### Spread & Outliers (`spending_amount`)")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(y=filtered_df["spending_amount"], ax=ax, color="#d95f02")
        ax.set_title("Spending Amount Boxplot", fontweight="bold")
        st.pyplot(fig)

with tab2:
    st.markdown("### Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 4))
    numeric_df = filtered_df[["age", "items_purchased", "spending_amount"]]
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Pearson Correlation Matrix", fontweight="bold")
    st.pyplot(fig)

st.divider()
st.dataframe(filtered_df.head(10), use_container_width=True)

# Execution command comment:
# cd Week_02_Visualization_Streamlit
# streamlit run day_13_advanced_streamlit_plots.py




