"""
Day 14: Week 2 Project - Production E-Commerce EDA & Insights App
Core Topics: Data Cleaning, Skewness, Outliers (IQR), Correlation, and Streamlit Embeddings
Execution Command: streamlit run day_14_production_capstone_dashboard.py
"""

import matplotlib.pyplot as plt    #creating figures
import numpy as np                 # calculations and synthetic data generation
import pandas as pd                # DataFrame manipulation, filtering, and aggregation
import seaborn as sns              # advanced statistical plots (KDE, Boxplots, Heatmaps)
import streamlit as st             # converting Python script into an interactive web app

# ----1. Page Setup (Must be the very first Streamlit command)----
st.set_page_config(
    page_title="E-Commerce Analytics Hub",
    page_icon="🛒",
    layout="wide",                   # full screen width
    initial_sidebar_state="expanded" # left sidebar panel open by default
)

# Apply global styling theme for Matplotlib/Seaborn graphics
sns.set_theme(
    style="whitegrid", palette= ["#2A9D8F", "#E76F51", "#3F27D6", "#E9C46A"]
) 

# ----2. Cached Data Generator & Cleaning Pipeline----
@st.cache_data                    # Cache dataset in RAM to prevent reload on every user interaction
def generate_dataset():
    np.random.seed(42)            # reproducible synthetic data
    n_samples = 2000              # transaction records
    
    cities = ["Lahore", "Karachi", "Islamabad", "Faisalabad", "Rawalpindi"]
    categories = ["Electronics", "Fashion", "Home & Kitchen", "Beauty"]
    
    # Generate synthetic DataFrame
    data = pd.DataFrame(
        {
            "order_id": np.arange(10001, 10001 + n_samples),  # Unique order IDs
            "city": np.random.choice(
                cities, size=n_samples, p=[0.35, 0.30, 0.15, 0.10, 0.10]
            ),                                                # City breakdown
            "category": np.random.choice(
                categories, size=n_samples
            ),                                                # Product category
            "age": np.random.normal(36, 10, size=n_samples)
            .clip(18, 70)
            .round(1),                                        # Customer age
            "items_purchased": np.random.randint(
                1, 9, size=n_samples),                        # Quantity bought
            "order_value": (
                np.random.exponential(scale=2200, size=n_samples) + 400
            ).round(
                2
            ),                                                # Right-skewed monetary value
        }
    ) 

     # Inject 3% synthetic missing values in order_value
    missing_indices = np.random.choice(
         n_samples, size=int(n_samples * 0.03), replace=False
    )
    data.loc[missing_indices, "order_value"] = np.nan

    # Impute missing values using Median (Outlier-robust)
    median_val = data["order_value"].median()
    data["order_value"] = data["order_value"].fillna(median_val)

    return data


# Load cleaned dataset into memory
df = generate_dataset()

# ----3. Header & Dashboard Context----
st.title("🛒 E-Commerce Production EDA & Insights Hub")
st.markdown(
    "Interactive exploratory data analysis platform"
)
st.divider()

# ----4. Sidebar Controls (Dynamic Filtering)----
st.sidebar.header("Executive Filter Controls")

selected_city = st.sidebar.selectbox(
    "Filter by Customer City:",
    options=["All Cities"] + sorted(df["city"].unique().tolist()),
)

selected_category = st.sidebar.selectbox(
    "Filter by Product Category:",
    options=["All Categories"] + sorted(df["category"].unique().tolist()),
)

# Apply dynamic filters on DataFrame copy
filtered_df = df.copy()
if selected_city != "All Cities":
    filtered_df = filtered_df[filtered_df["city"] == selected_city]
if selected_category != "All Categories":
    filtered_df = filtered_df[filtered_df["category"] == selected_category]
    
# ----5. Executive KPI Metric Cards----
calc_revenue = filtered_df["order_value"].sum()            # Total revenue
calc_orders = len(filtered_df)                             # Order count
calc_aov = (
    calc_revenue / calc_orders if calc_orders > 0 else 0
)  # Average Order Value

kpi1, kpi2, kpi3 = st.columns(3)                           # Split screen into 3 columns
kpi1.metric("Total Revenue", f"PKR {calc_revenue:,.2f}")
kpi2.metric("Total Orders", f"{calc_orders:,}")
kpi3.metric("Average Order Value (AOV)", f"PKR {calc_aov:,.2f}")

st.divider()

# 6. Tabbed Analytical Navigation System
tab1, tab2, tab3 = st.tabs(
    [
        " Distributions & Skewness",
        " Outlier Detection (IQR)",
        " Correlation & Data Grid",
    ]
)

# --- TAB 1: Distribution & Skewness Diagnosis ---
with tab1:
    st.subheader("Statistical Distribution & Skewness Diagnosis")
    col_a, col_b = st.columns(2)                  # Side-by-side plot layout

    # Age Distribution (Normal / Symmetric)
    with col_a:
        st.markdown("#### Customer Age Distribution")
        age_skew = filtered_df["age"].skew()       # Calculate skewness score
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(
            filtered_df["age"], kde=True, ax=ax, color="#2A9D8F"
        )                                           # Histogram with KDE
        ax.set_title(f"Age (Skew: {age_skew:.2f})", fontweight="bold")
        st.pyplot(fig)  # Render plot in Streamlit

    # Order Value Distribution (Right-Skewed)
    with col_b:
        st.markdown("#### Order Value Distribution")
        spend_skew = filtered_df["order_value"].skew()  # Calculate skewness score
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(
            filtered_df["order_value"], kde=True, ax=ax, color="#E76F51"
        )  # Histogram with KDE
        ax.set_title(f"Order Value (Skew: {spend_skew:.2f})", fontweight="bold")
        st.pyplot(fig)  # Render plot in Streamlit

# --- TAB 2: Outlier Detection via Tukey's Fences (IQR) ---
with tab2:
    st.subheader("Outlier Detection & Tukey's Fence Thresholds")

    # IQR Math Calculations
    q1 = filtered_df["order_value"].quantile(0.25)  # 25th percentile
    q3 = filtered_df["order_value"].quantile(0.75)  # 75th percentile
    iqr = q3 - q1  # Interquartile range
    upper_fence = q3 + (1.5 * iqr)  # Upper threshold cutoff formula

    outliers = filtered_df[filtered_df["order_value"] > upper_fence]
    outlier_count = len(outliers)
    outlier_pct = (
        (outlier_count / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0
    )

    st.info(
        f"**IQR Analysis:** Q1 = PKR {q1:,.2f} | Q3 = PKR {q3:,.2f} | **Upper Cutoff = PKR {upper_fence:,.2f}**\n\n"
        f"Identified **{outlier_count} Outlier Transactions** ({outlier_pct:.2f}% of filtered sample)."
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(x=filtered_df["order_value"], ax=ax, color="#3F27D6")
    ax.axvline(
        upper_fence, color="red", linestyle="--", label=f"Upper Cutoff ({upper_fence:,.0f})"
    )  # Red dashed threshold line
    ax.set_title("Order Value Boxplot with Outlier Cutoff Line", fontweight="bold")
    ax.legend(loc="upper right")
    st.pyplot(fig)

# --- TAB 3: Correlation Matrix & Interactive Data Grid ---
with tab3:
    st.subheader("Feature Correlation & Raw Dataset Grid")
    col_c, col_d = st.columns([1, 1])

    with col_c:
        st.markdown("#### Pearson Correlation Matrix")
        numeric_cols = ["age", "items_purchased", "order_value"]
        corr_matrix = filtered_df[numeric_cols].corr()  # Pairwise correlation

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            corr_matrix, annot=True, fmt=".2f", cmap="YlGnBu", vmin=-1, vmax=1, ax=ax
        )                                              # Correlation Heatmap
        ax.set_title("Numeric Correlation Heatmap", fontweight="bold")
        st.pyplot(fig)

    with col_d:
        st.markdown("#### Dynamic Transaction Data View")
        st.dataframe(
            filtered_df, use_container_width=True, height=300
        )                                              # Interactive data grid

st.divider()



# to run this dashboard
# cd D:\Projects\ML_Engineering_Sprint\Week_02_Visualization_Streamlit
# streamlit run day_14_production_capstone_dashboard.py
