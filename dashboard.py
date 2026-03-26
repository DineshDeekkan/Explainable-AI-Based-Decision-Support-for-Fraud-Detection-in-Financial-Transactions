import streamlit as st
from kafka import KafkaConsumer
import json
import pandas as pd
import matplotlib.pyplot as plt

st.title("🚨 Real-Time Financial Fraud Detection Dashboard")

# Kafka consumer
consumer = KafkaConsumer(
    "transactions_blocked",
    "transactions_approved",
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    auto_offset_reset="latest",
    enable_auto_commit=True,
    group_id="fraud-dashboard"
)

transactions = []

placeholder = st.empty()

for message in consumer:
    txn = message.value
    transactions.append(txn)

    df = pd.DataFrame(transactions)

    total = len(df)
    fraud = len(df[df["predicted_label"] == "FRAUD"])
    genuine = len(df[df["predicted_label"] == "GENUINE"])

    fraud_rate = (fraud / total) * 100 if total > 0 else 0

    with placeholder.container():

        st.metric("Total Transactions", total)
        st.metric("Fraud Detected", fraud)
        st.metric("Genuine Transactions", genuine)
        st.metric("Fraud Rate (%)", round(fraud_rate, 2))

        st.subheader("Fraud Probability Distribution")

        fig, ax = plt.subplots()
        ax.hist(df["fraud_probability"], bins=20)
        st.pyplot(fig)

        st.subheader("Recent Transactions")
        st.dataframe(df.tail(10))