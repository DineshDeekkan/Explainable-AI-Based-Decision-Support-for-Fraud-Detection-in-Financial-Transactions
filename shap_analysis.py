from kafka import KafkaConsumer, KafkaProducer
import json
import matplotlib.pyplot as plt
import pandas as pd
import shap
import joblib

print("Starting SHAP explainer for a fraud transaction...")

# Load trained model and encoders
model = joblib.load("fraud_model.pkl")
location_encoder = joblib.load("location_encoder.pkl")
device_encoder = joblib.load("device_encoder.pkl")

# Kafka consumer for blocked transactions
consumer = KafkaConsumer(
    "transactions_blocked",
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    auto_offset_reset="latest",
    group_id="shap-analysis-group"
)

# Kafka producer to send SHAP insights
producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

print("Waiting for a fraud transaction...")

# Create SHAP explainer
explainer = shap.TreeExplainer(model)

for message in consumer:
    txn = message.value
    print("\nFraud transaction received:", txn)

    try:
        # Convert to dataframe
        df = pd.DataFrame([txn])

        # Encode categorical features
        df["location"] = location_encoder.transform(df["location"])
        df["device"] = device_encoder.transform(df["device"])

        X = df[["amount", "location", "device", "risk_score"]]

        # SHAP explanation
        shap_values = explainer.shap_values(X)

        print("Generating SHAP explanation...")

        if isinstance(shap_values, list):
            shap_val = shap_values[1][0]
        else:
            shap_val = shap_values[0][:, 1]

        feature_impacts = list(zip(X.columns, shap_val))
        feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)

        reasons = []
        for feature, val in feature_impacts[:3]:
            if val > 0:
                reasons.append(f"{feature} increased fraud risk")
            else:
                reasons.append(f"{feature} reduced fraud risk")

        summary = "Fraud due to " + ", ".join(reasons[:2])

        # 🚀 Send SHAP data
        producer.send("shap_insights", {
            "txn_id": txn["txn_id"],
            "features": [f for f, _ in feature_impacts],
            "impacts": [float(v) for _, v in feature_impacts],
            "summary": summary
        })

        print("SHAP Sent:", summary)
        print("-" * 60)

    except Exception as e:
        print("SHAP Error:", e)