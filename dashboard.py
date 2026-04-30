import streamlit as st
import psycopg2
import pandas as pd
import plotly.express as px

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(page_title="Fraud Intelligence Dashboard", layout="wide")

st.title("🚨 XAI-Based Fraud Intelligence Dashboard")

# =============================
# DB CONNECTION
# =============================
@st.cache_resource
def get_connection():
    return psycopg2.connect(
        host="localhost",
        database="fy project",
        user="postgres",
        password="Dinesh"
    )

conn = get_connection()

# =============================
# LOAD DATA (SEPARATE TABLES)
# =============================
@st.cache_data(ttl=5)
def load_data():
    transactions = pd.read_sql("SELECT * FROM transactions", conn)
    approved = pd.read_sql("SELECT * FROM transactions_approved", conn)
    blocked = pd.read_sql("SELECT * FROM transactions_blocked", conn)
    metrics = pd.read_sql("SELECT * FROM model_metrics ORDER BY created_at DESC LIMIT 1", conn)
    shap = pd.read_sql("SELECT * FROM shap_insights", conn)
    return transactions, approved, blocked, metrics, shap

transactions_df, approved_df, blocked_df, metrics, shap_df = load_data()

# =============================
# SIDEBAR FILTERS
# =============================
st.sidebar.header("🔍 Filters")

label_filter = st.sidebar.selectbox("Transaction Type", ["ALL", "FRAUD", "GENUINE"])

# =============================
# DATA SELECTION LOGIC (IMPORTANT CHANGE)
# =============================
if label_filter == "ALL":
    df = transactions_df
elif label_filter == "FRAUD":
    df = blocked_df
else:
    df = approved_df

# Dynamic filters based on selected data
location_filter = st.sidebar.multiselect("Location", df["location"].unique())
device_filter = st.sidebar.multiselect("Device", df["device"].unique())

amount_range = st.sidebar.slider(
    "Amount Range",
    int(df["amount"].min()),
    int(df["amount"].max()),
    (int(df["amount"].min()), int(df["amount"].max()))
)

# Apply filters
filtered_df = df.copy()

if location_filter:
    filtered_df = filtered_df[filtered_df["location"].isin(location_filter)]

if device_filter:
    filtered_df = filtered_df[filtered_df["device"].isin(device_filter)]

filtered_df = filtered_df[
    (filtered_df["amount"] >= amount_range[0]) &
    (filtered_df["amount"] <= amount_range[1])
]

# =============================
# METRICS
# =============================
total = len(filtered_df)
fraud = len(filtered_df[filtered_df["predicted_label"] == "FRAUD"])
genuine = len(filtered_df[filtered_df["predicted_label"] == "GENUINE"])
fraud_rate = (fraud / total * 100) if total else 0

st.markdown("## 📊 Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Transactions", total)
c2.metric("Fraud", fraud)
c3.metric("Genuine", genuine)
c4.metric("Fraud %", round(fraud_rate, 2))

st.markdown("---")

# =============================
# MODEL METRICS
# =============================
st.markdown("## 📈 Model Performance")

if not metrics.empty:
    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", metrics["accuracy"].values[0])
    m2.metric("Precision", metrics["precision"].values[0])
    m3.metric("Recall", metrics["recall"].values[0])

st.markdown("---")

# =============================
# ADVANCED ANALYTICS
# =============================
st.markdown("## 📊 Advanced Analytics")

col1, col2 = st.columns(2)

# Fraud vs Genuine Distribution
with col1:
    fig = px.pie(filtered_df, names="predicted_label", title="Fraud vs Genuine")
    st.plotly_chart(fig, use_container_width=True)

# Fraud Probability Distribution (NEW)
with col2:
    if "fraud_probability" in filtered_df.columns:
        prob_fig = px.histogram(
            filtered_df,
            x="fraud_probability",
            nbins=40,
            title="Fraud Probability Distribution"
        )
        st.plotly_chart(prob_fig, use_container_width=True)

# =============================
# AMOUNT ANALYSIS (NEW)
# =============================
st.markdown("### 💰 Transaction Amount Insights")

col3, col4 = st.columns(2)

# Boxplot for anomaly detection
with col3:
    box_fig = px.box(
        filtered_df,
        x="predicted_label",
        y="amount",
        title="Amount Distribution by Class"
    )
    st.plotly_chart(box_fig, use_container_width=True)

# Fraud by device
with col4:
    device_fig = px.bar(
        filtered_df[filtered_df["predicted_label"] == "FRAUD"],
        x="device",
        title="Fraud by Device"
    )
    st.plotly_chart(device_fig, use_container_width=True)

# =============================
# TIME SERIES
# =============================
st.markdown("### 📈 Live Transaction Trend")

filtered_df["time"] = pd.to_datetime(filtered_df["timestamp"], unit="s")

trend = filtered_df.groupby(pd.Grouper(key="time", freq="1min")).size().reset_index(name="count")

trend_fig = px.line(trend, x="time", y="count", title="Transactions Over Time")
st.plotly_chart(trend_fig, use_container_width=True)

st.markdown("---")

# =============================
# LIVE TABLE (IMPORTANT CHANGE)
# =============================
st.markdown("## 🔴 Live Transactions Feed")

# Show recent transactions based on selection
st.dataframe(filtered_df.sort_values("timestamp", ascending=False).head(20))

# =============================
# ALERTS (ONLY FOR FRAUD VIEW)
# =============================
if label_filter == "FRAUD":
    st.markdown("## 🚨 Fraud Alerts")

    if not filtered_df.empty:
        for _, row in filtered_df.head(5).iterrows():
            st.error(f"🚨 Fraud TXN: {row['txn_id']} | ₹{row['amount']} | {row['location']}")

st.markdown("---")

# =============================
# TRANSACTION EXPLORER
# =============================
st.markdown("## 🔎 Transaction Explorer")

if not filtered_df.empty:
    selected_txn = st.selectbox("Select Transaction", filtered_df["txn_id"])

    txn_data = filtered_df[filtered_df["txn_id"] == selected_txn]
    st.dataframe(txn_data)

    st.markdown("### 🔍 SHAP Explanation")

    txn_shap = shap_df[shap_df["txn_id"] == selected_txn]

    if not txn_shap.empty:
        fig = px.bar(
            txn_shap.sort_values("impact", ascending=False).head(5),
            x="impact",
            y="feature_name",
            orientation="h",
            title="Top Feature Impacts"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info(txn_shap["summary"].iloc[0])
    else:
        st.warning("No SHAP data available")