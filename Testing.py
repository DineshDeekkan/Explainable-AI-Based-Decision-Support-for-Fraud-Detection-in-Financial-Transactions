import streamlit as st
from kafka import KafkaConsumer
import json
import pandas as pd
import matplotlib.pyplot as plt

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(page_title="Fraud Dashboard", layout="wide")

# =============================
# 🔥 PREMIUM CSS
# =============================
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}

.metric-card {
    background: linear-gradient(135deg, #1f2937, #111827);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.5);
}

.metric-title {
    font-size: 16px;
    color: #9ca3af;
}

.metric-value {
    font-size: 28px;
    font-weight: bold;
    color: #00ffcc;
}

.section-title {
    font-size: 22px;
    margin-top: 10px;
    margin-bottom: 10px;
    color: #f9fafb;
}

.card {
    background: #1f2937;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>🚨 Fraud Detection Dashboard</h1>", unsafe_allow_html=True)

# =============================
# KAFKA
# =============================
consumer = KafkaConsumer(
    "transactions_blocked",
    "transactions_approved",
    "model_metrics",
    "shap_insights",
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    auto_offset_reset="latest",
    group_id="fraud-dashboard"
)

transactions = []
metrics_data = {}
shap_data = {}

placeholder = st.empty()

# =============================
# STREAM LOOP
# =============================
for message in consumer:
    topic = message.topic
    data = message.value

    if topic in ["transactions_blocked", "transactions_approved"]:
        if isinstance(data, dict) and "predicted_label" in data:
            transactions.append(data)
            if len(transactions) > 1000:
                transactions.pop(0)

    elif topic == "model_metrics":
        metrics_data = data

    elif topic == "shap_insights":
        shap_data = data

    df = pd.DataFrame(transactions)

    if not df.empty and "predicted_label" in df.columns:
        total = len(df)
        fraud = len(df[df["predicted_label"] == "FRAUD"])
        genuine = len(df[df["predicted_label"] == "GENUINE"])
        fraud_rate = (fraud / total * 100) if total else 0
    else:
        total, fraud, genuine, fraud_rate = 0, 0, 0, 0

    with placeholder.container():

        # =============================
        # 🔥 METRIC CARDS
        # =============================
        st.markdown("<div class='section-title'>📊 Overview</div>", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)

        def card(title, value):
            return f"""
            <div class="metric-card">
                <div class="metric-title">{title}</div>
                <div class="metric-value">{value}</div>
            </div>
            """

        c1.markdown(card("Total", total), unsafe_allow_html=True)
        c2.markdown(card("Fraud", fraud), unsafe_allow_html=True)
        c3.markdown(card("Genuine", genuine), unsafe_allow_html=True)
        c4.markdown(card("Fraud %", round(fraud_rate, 2)), unsafe_allow_html=True)

        # =============================
        # 📈 MODEL + DISTRIBUTION
        # =============================
        left, right = st.columns(2)

        with left:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>📈 Model Performance</div>", unsafe_allow_html=True)

            if metrics_data:
                st.write(f"Accuracy: **{metrics_data.get('accuracy',0)}**")
                st.write(f"Precision: **{metrics_data.get('precision',0)}**")
                st.write(f"Recall: **{metrics_data.get('recall',0)}**")

                TP = metrics_data.get("TP", 0)
                TN = metrics_data.get("TN", 0)
                FP = metrics_data.get("FP", 0)
                FN = metrics_data.get("FN", 0)

                st.info(f"TP:{TP} | TN:{TN} | FP:{FP} | FN:{FN}")

            else:
                st.warning("Waiting for metrics...")

            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>📊 Probability Distribution</div>", unsafe_allow_html=True)

            if not df.empty and "fraud_probability" in df.columns:
                fig, ax = plt.subplots()
                ax.hist(df["fraud_probability"], bins=20)
                ax.set_facecolor("#1f2937")
                fig.patch.set_facecolor("#1f2937")
                st.pyplot(fig)
            else:
                st.warning("No data yet")

            st.markdown("</div>", unsafe_allow_html=True)

        # =============================
        # 🔍 SHAP + TABLE
        # =============================
        left, right = st.columns(2)

        with left:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>🔍 SHAP Insights</div>", unsafe_allow_html=True)

            if shap_data:
                features = shap_data.get("features", [])
                impacts = shap_data.get("impacts", [])

                if features:
                    fig, ax = plt.subplots()
                    ax.barh(features[:5], impacts[:5])
                    ax.set_facecolor("#1f2937")
                    fig.patch.set_facecolor("#1f2937")
                    st.pyplot(fig)

                st.success(shap_data.get("summary", ""))

            else:
                st.warning("Waiting for SHAP...")

            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>📋 Transactions</div>", unsafe_allow_html=True)

            if not df.empty:
                st.dataframe(df.tail(10), use_container_width=True)
            else:
                st.warning("No transactions")

            st.markdown("</div>", unsafe_allow_html=True)