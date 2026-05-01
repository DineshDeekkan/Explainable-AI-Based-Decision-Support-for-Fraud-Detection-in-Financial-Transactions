import streamlit as st
from streamlit_autorefresh import st_autorefresh
import psycopg2
import pandas as pd
import plotly.express as px
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from datetime import datetime

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(page_title="Fraud Intelligence Dashboard", layout="wide")

st.title("🚨 XAI-Based Fraud Intelligence Dashboard")

# Auto refresh every 5 seconds (5000 ms)
st_autorefresh(interval=5000, key="datarefresh")

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
#@st.cache_data(ttl=5)
def load_data():
    transactions = pd.read_sql("SELECT * FROM transactions_final", conn)
    approved = pd.read_sql("SELECT * FROM transactions_approved", conn)
    blocked = pd.read_sql("SELECT * FROM transactions_blocked", conn)
    metrics = pd.read_sql("SELECT * FROM model_metrics ORDER BY created_at DESC LIMIT 1", conn)
    shap = pd.read_sql("SELECT * FROM shap_insights", conn)
    model_comparison = pd.read_sql("SELECT * FROM model_comparison", conn)

    return transactions, approved, blocked, metrics, shap, model_comparison

transactions_df, approved_df, blocked_df, metrics, shap_df, model_df = load_data()

placeholder = st.empty()

with placeholder.container():
    # =============================
    # SIDEBAR FILTERS
    # =============================
    st.sidebar.header("🔍 Filters")

    label_filter = st.sidebar.selectbox("Transaction Type", ["ALL", "FRAUD", "GENUINE"])

# =============================
# DATA SELECTION LOGIC (IMPORTANT CHANGE)
# =============================
df = transactions_df.copy()

if label_filter == "FRAUD":
    df = df[df["predicted_label"] == "FRAUD"]
elif label_filter == "GENUINE":
    df = df[df["predicted_label"] == "GENUINE"]

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
fraud = (filtered_df["predicted_label"] == "FRAUD").sum()
genuine = (filtered_df["predicted_label"] == "GENUINE").sum()

# safety check
if fraud + genuine != total:
    genuine = total - fraud

fraud_rate = (fraud / total * 100) if total else 0

st.success("🟢 Live Data Streaming Active")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

st.markdown("## 📊 Transaction Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Transactions", total)
c2.metric("Fraud", fraud)
c3.metric("Genuine", genuine)
c4.metric("Fraud %", round(fraud_rate, 2))

st.markdown("---")

# =============================
# RISK LEVEL INDICATOR
# =============================
st.markdown("#### ⚠️ System Risk Level")

if "fraud_probability" in filtered_df.columns and not filtered_df.empty:

    avg_risk = filtered_df["fraud_probability"].mean()

    if avg_risk > 0.7:
        st.error("🔴 High Risk Level")
    elif avg_risk > 0.4:
        st.warning("🟡 Moderate Risk Level")
    else:
        # fallback
        st.success("🟢 Low Risk Level")

else:
    # fallback
    st.success("🟢 Low Risk Level")

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
# MODEL COMPARISON DASHBOARD
# =============================
st.markdown("## 🤖 Model Comparison")

if not model_df.empty:

    st.dataframe(model_df)

    fig = px.bar(
        model_df,
        x="model_name",
        y=["accuracy", "precision", "recall"],
        barmode="group",
        title="Model Performance Comparison"
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("No model comparison data available")

# =============================
# CONFUSION MATRIX
# =============================
import plotly.figure_factory as ff

st.markdown("### 📉 Confusion Matrix")

if not metrics.empty and all(col in metrics.columns for col in [
    "tp", "tn", "fp", "fn"
]):

    tp = metrics["tp"].values[0]
    tn = metrics["tn"].values[0]
    fp = metrics["fp"].values[0]
    fn = metrics["fn"].values[0]

    z = [
        [tn, fp],
        [fn, tp]
    ]

    x = ["Predicted Genuine", "Predicted Fraud"]
    y = ["Actual Genuine", "Actual Fraud"]

    fig = ff.create_annotated_heatmap(
        z,
        x=x,
        y=y,
        colorscale="Blues"
    )

    st.plotly_chart(fig, use_container_width=True)

    # =============================
    # INTERPRETATION (VERY IMPORTANT)
    # =============================
    st.info(f"""
    🔍 Interpretation:
    
    ✔ True Positives (Fraud correctly detected): {tp}  
    ✔ True Negatives (Genuine correctly detected): {tn}  
    ⚠ False Positives (Wrongly flagged as fraud): {fp}  
    🚨 False Negatives (Missed fraud cases): {fn}  
    
    """)

else:
    st.warning("Confusion matrix data not available")

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

# =============================
# FRAUD BY LOCATION
# =============================
st.markdown("### 🌍 Fraud by Location")

fraud_df = filtered_df[filtered_df["predicted_label"] == "FRAUD"]

if not fraud_df.empty:

    location_counts = fraud_df["location"].value_counts().reset_index()
    location_counts.columns = ["location", "count"]

    fig = px.bar(
        location_counts,
        x="location",
        y="count",
        title="Fraud Distribution by Location"
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("No fraud data available for selected filters")
st.markdown("---")

# =============================
# LIVE TABLE (IMPORTANT CHANGE)
# =============================
st.markdown("## 🔴 Live Transactions Monitoring")

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

    # Sort by impact
    waterfall_df = txn_shap.sort_values("impact", ascending=False).head(8)

    # Create direction column
    waterfall_df["direction"] = waterfall_df["impact"].apply(
        lambda x: "Increase Fraud Risk" if x > 0 else "Decrease Fraud Risk"
    )

    fig = px.bar(
        waterfall_df,
        x="impact",
        y="feature_name",
        orientation="h",
        color="direction",
        title="Feature Contribution to Prediction"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    🔍 Positive values increase fraud risk, while negative values decrease it.
    """)

else:
    st.warning("No SHAP data available")

# =============================
# GLOBAL FEATURE IMPORTANCE
# =============================
st.markdown("## 🧠 Global Feature Importance (XAI)")

if not shap_df.empty:

    global_shap = (
        shap_df.groupby("feature_name")["impact"]
        .apply(lambda x: x.abs().mean())
        .reset_index()
        .sort_values(by="impact", ascending=False)
        .head(10)
    )

    col1, col2 = st.columns(2)

    # Bar chart
    with col1:
        fig = px.bar(
            global_shap,
            x="impact",
            y="feature_name",
            orientation="h",
            title="Top Influencing Features"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Pie chart
    with col2:
        fig2 = px.pie(
            global_shap,
            values="impact",
            names="feature_name",
            title="Feature Contribution Share"
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Insight
    top_feature = global_shap.iloc[0]["feature_name"]

    st.info(f"""
    🔍 Insight:
    The most important feature influencing fraud detection is **{top_feature}**.
    """)

else:
    st.warning("No SHAP data available for global analysis")

# =============================
# SHAP SUMMARY DISTRIBUTION
# =============================
st.markdown("## 📊 SHAP Summary Distribution")

if not shap_df.empty:

    # Take top features only
    top_features = (
        shap_df.groupby("feature_name")["impact"]
        .apply(lambda x: x.abs().mean())
        .sort_values(ascending=False)
        .head(5)
        .index.tolist()
    )

    summary_df = shap_df[shap_df["feature_name"].isin(top_features)]

    fig = px.strip(
        summary_df,
        x="impact",
        y="feature_name",
        color="feature_name",
        title="SHAP Impact Distribution Across Transactions"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    🔍 Each dot represents a transaction.
    The spread shows how feature impact varies across different cases.
    """)

else:
    st.warning("No SHAP data available")

# =============================
# FEATURE INTERACTION HEATMAP
# =============================
st.markdown("## 🎯 Feature Interaction Analysis")

if not shap_df.empty:

    # Pivot data
    pivot_df = shap_df.pivot_table(
        index="txn_id",
        columns="feature_name",
        values="impact",
        fill_value=0
    )

    # Correlation
    corr = pivot_df.corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        title="Feature Interaction Heatmap"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    🔍 This heatmap shows how features interact with each other.
    Strong correlations indicate combined influence on fraud detection.
    """)

else:
    st.warning("No data for interaction analysis")

# =============================
# FRAUD REPORT GENERATION
# =============================
def generate_pdf(total, fraud, genuine, fraud_rate):

    doc = SimpleDocTemplate("fraud_report.pdf")
    styles = getSampleStyleSheet()

    content = []

    # =============================
    # TITLE
    # =============================
    content.append(Paragraph("Fraud Intelligence Report", styles["Title"]))
    content.append(Spacer(1, 12))

    content.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    content.append(Spacer(1, 12))

    # =============================
    # SUMMARY TABLE
    # =============================
    content.append(Paragraph("Transaction Summary", styles["Heading2"]))
    content.append(Spacer(1, 10))

    table_data = [
        ["Metric", "Value"],
        ["Total Transactions", total],
        ["Fraud Transactions", fraud],
        ["Genuine Transactions", genuine],
        ["Fraud Rate (%)", f"{fraud_rate:.2f}"]
    ]

    table = Table(table_data)

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("PADDING", (0, 0), (-1, -1), 8)
    ]))

    content.append(table)
    content.append(Spacer(1, 20))

    # =============================
    # INSIGHTS
    # =============================
    content.append(Paragraph("Key Insights", styles["Heading2"]))
    content.append(Spacer(1, 10))

    insight_text = f"""
    The system analyzed {total} transactions. 
    Out of these, {fraud} were detected as fraudulent and {genuine} as genuine.

    The fraud rate is {fraud_rate:.2f}%, indicating the overall risk level in the system.

    The dashboard uses Explainable AI (XAI) techniques to provide transparency 
    and interpretability in fraud detection decisions.
    """

    content.append(Paragraph(insight_text, styles["Normal"]))
    content.append(Spacer(1, 20))

    # =============================
    # FOOTER
    # =============================
    content.append(Paragraph("Generated by XAI Fraud Detection System", styles["Italic"]))

    doc.build(content)

    return "fraud_report.pdf"

# =============================
# EXPORT REPORT
# =============================
st.markdown("#### 📑 Export Report")

if st.button("Generate PDF Report"):

    file_path = generate_pdf(total, fraud, genuine, fraud_rate)

    with open(file_path, "rb") as f:
        st.download_button(
            label="Download Report",
            data=f,
            file_name="fraud_report.pdf",
            mime="application/pdf"
        )

