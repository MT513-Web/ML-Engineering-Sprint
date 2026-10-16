"""
Day 12: Streamlit Fundamentals & Interactive Data Dashboards
Core Topics: Script Execution Model, Sidebar Widgets, KPI Metrics, Dynamic Data Filtering
"""

import numpy as np
import pandas as pd
import streamlit as st

# ----1. Page Configuration (Must be the first Streamlit command)----
st.set_page_config(
    page_title="E-Commerce Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Header & Overview
st.title("📊 Production E-Commerce Analytics Dashboard")
st.markdown(
    "Interactive business dashboard demonstrating **dynamic data filtering** and **KPI aggregation** using Streamlit."
)
st.divider()

# 3. Cached Data Generation (Streamlit runs script top-to-bottom on every interaction)
@st.cache_data
def load_transaction_data():
    np.random.seed(42)
    n = 1500
    cities = ["Lahore", "Karachi", "Islamabad", "Faisalabad", "Rawalpindi"]
    tiers = ["Bronze", "Silver", "Gold"]
    
    data = pd.DataFrame(
        {
            "order_id": np.arange(1001, 1001 + n),
            "city": np.random.choice(cities, size=n, p=[0.35, 0.30, 0.15, 0.10, 0.10]),
            "membership_tier": np.random.choice(tiers, size=n, p=[0.50, 0.35, 0.15]),
            "order_value": (np.random.exponential(scale=1800, size=n) + 300).round(2),
            "items_count": np.random.randint(1, 8, size=n),
        }
    )
    return data

df =  load_transaction_data()

# 4. Sidebar Controls (User Input Widgets)
st.sidebar.header("Filter Controls")

selected_city = st.sidebar.selectbox(
    "Select Target City:",
    options=["All Cities"] + sorted(df["city"].unique().tolist()),
    index=0,
) 

min_spend,  max_spend = st.sidebar.slider(
    "Order Value Range (PKR):",
    min_value=float(df["order_value"].min()),
    max_value=float(df["order_value"].max()),
    value=(float(df["order_value"].min()), float(df["order_value"].max())),
)

# 5. Dynamic Data Filtering
filtered_df = df.copy()

if selected_city != "All Cities":
    filtered_df = filtered_df[filtered_df["city"]== selected_city]
    
    filtered_df = filtered_df[
    (filtered_df["order_value"] >= min_spend) & (filtered_df["order_value"] <= max_spend)
]
    
# 6. KPI Metric Cards
total_revenue = filtered_df["order_value"].sum()
total_orders = len(filtered_df)
avg_order_value = filtered_df["order_value"].mean() if total_orders > 0 else 0

col1, col2, col3 = st.columns(3)
col1.metric("Total Revenue", f"PKR {total_revenue:,.2f}")
col2.metric("Total Orders", f"{total_orders:,}")
col3.metric("Average Order Value (AOV)", f"PKR {avg_order_value:,.2f}")

st.divider()

# 7. Interactive Data Preview
st.subheader("📋 Filtered Transactions Data")
st.dataframe(filtered_df, use_container_width=True)






