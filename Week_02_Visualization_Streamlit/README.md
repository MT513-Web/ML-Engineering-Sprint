# Week 2: E-Commerce Production EDA & Analytics Dashboard

An interactive, production-grade Streamlit web application designed for exploratory data analysis, statistical distribution diagnosis, outlier detection, and executive KPI reporting on e-commerce transaction data.

## Live Demo
* **Streamlit Community Cloud:** [E-Commerce Analytics Hub](https://ecommerceanalyticsdashboard-2dqe3dbtdwxveq5h4bugdx.streamlit.app)

## Key Features & Capabilities
1. **Executive Filter Controls:** Real-time sidebar filtering by Customer City (Faisalabad, Islamabad, Karachi, Lahore, Rawalpindi) and Product Category (Beauty, Electronics, Fashion, Home & Kitchen)[cite: 36, 37].
2. **Dynamic KPI Metrics:** Automatic calculation of Total Revenue (PKR), Total Orders, and Average Order Value (AOV) based on active filters[cite: 36, 37, 38].
3. **Distributions & Skewness Analysis:** Visual diagnosis and skewness tracking for Customer Age and Order Value distributions[cite: 36].
4. **Outlier Detection (IQR):** Tukey's Fence thresholding and boxplot visualization to identify anomalous transaction values and outlier counts[cite: 37].
5. **Correlation & Data Grid:** Interactive Pearson Correlation Heatmap for numeric features (age, items purchased, order value) alongside a dynamic transaction data view[cite: 38].

## Project Structure
* `day_08_production_pandas_vectorization.py` - High-performance vectorized pandas pipeline
* `day_09_production_data_cleaning.py` - Missing value imputation and data sanitization
* `day_10_production_groupby_aggregations.py` - Multi-index aggregations and pivots
* `day_11_production_eda_visualization.py` - Statistical distribution scripts
* `day_12_production_streamlit_basics.py` - Streamlit dashboard foundational scripts
* `day_13_advanced_streamlit_plots.py` - Embedded Matplotlib charts in Streamlit
* `day_14_production_capstone_dashboard.py` - Main unified E-commerce analytics app
* `eda_step1_distribution.png`, `eda_step2_boxplots.png`, `eda_step3_correlation.png` - Exported EDA visual artifacts

## Run Locally
```bash
pip install -r requirements.txt
streamlit run day_14_production_capstone_dashboard.py