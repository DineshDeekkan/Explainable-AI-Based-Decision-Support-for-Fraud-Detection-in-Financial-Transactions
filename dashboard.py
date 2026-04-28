import streamlit as st
import psycopg2
import pandas as pd
import plotly.express as px

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(page_title="Fraud Intelligence Dashboard", layout="wide")

st.markdown("""
<style>
body { background-color: #0e1117; }
h1, h2, h3 { color: #ffffff; }
</style>
""", unsafe_allow_html=True)

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
# LOAD DATA
# =============================
@st.cache_data(ttl=5)
def load_data():
    df = pd.read_sql("SELECT * FROM transactions_approved UNION ALL SELECT * FROM transactions_blocked", conn)
    metrics = pd.read_sql("SELECT * FROM model_metrics ORDER BY created_at DESC LIMIT 1", conn)
    shap = pd.read_sql("SELECT * FROM shap_insights", conn)
    return df, metrics, shap

df, metrics, shap_df = load_data()

# =============================
# SIDEBAR FILTERS
# =============================
st.sidebar.header("🔍 Filters")

label_filter = st.sidebar.selectbox("Transaction Type", ["ALL", "FRAUD", "GENUINE"])

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

if label_filter != "ALL":
    filtered_df = filtered_df[filtered_df["predicted_label"] == label_filter]

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

avg_amount = filtered_df["amount"].mean() if total else 0
max_amount = filtered_df["amount"].max() if total else 0

st.markdown("## 📊 Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Transactions", total)
c2.metric("Fraud", fraud)
c3.metric("Genuine", genuine)
c4.metric("Fraud %", round(fraud_rate, 2))

c5, c6 = st.columns(2)
c5.metric("Avg Amount", round(avg_amount, 2))
c6.metric("Max Amount", round(max_amount, 2))

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
else:
    st.warning("No metrics available")

st.markdown("---")

# =============================
# INTERACTIVE CHARTS
# =============================
st.markdown("## 📊 Analytics")

col1, col2 = st.columns(2)

# Fraud Distribution
with col1:
    fig = px.pie(filtered_df, names="predicted_label", title="Fraud vs Genuine")
    st.plotly_chart(fig, use_container_width=True)

# Location Analysis
with col2:
    loc_fig = px.bar(
        filtered_df[filtered_df["predicted_label"] == "FRAUD"],
        x="location",
        title="Fraud by Location"
    )
    st.plotly_chart(loc_fig, use_container_width=True)

# Time Trend
st.markdown("### 📈 Transaction Trend")

filtered_df["time"] = pd.to_datetime(filtered_df["timestamp"], unit="s")

trend = filtered_df.groupby(pd.Grouper(key="time", freq="1min")).size().reset_index(name="count")

fig = px.line(trend, x="time", y="count", title="Transactions Over Time")
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# =============================
# ALERTS
# =============================
st.markdown("## 🚨 Live Alerts")

fraud_df = filtered_df[filtered_df["predicted_label"] == "FRAUD"]

if not fraud_df.empty:
    for _, row in fraud_df.tail(5).iterrows():
        st.error(f"🚨 Fraud transaction : {row['txn_id']}  |  Amount : ₹{row['amount']}  |  Location : {row['location']}")
else:
    st.success("No fraud alerts")

st.markdown("---")

# =============================
# TRANSACTION + SHAP DRILLDOWN
# =============================
st.markdown("## 🔎 Transaction Explorer")

if not filtered_df.empty:
    selected_txn = st.selectbox("Select Transaction", filtered_df["txn_id"])

    txn_data = filtered_df[filtered_df["txn_id"] == selected_txn]
    st.dataframe(txn_data)

    # SHAP
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
        st.warning("No SHAP data for this transaction")