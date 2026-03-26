from kafka import KafkaConsumer
import json
import matplotlib.pyplot as plt
import pandas as pd
import shap
import joblib

print("Starting SHAP explanation for a fraud transaction...")

# Load trained model and encoders
model = joblib.load("fraud_model.pkl")
location_encoder = joblib.load("location_encoder.pkl")
device_encoder = joblib.load("device_encoder.pkl")

# Kafka consumer
consumer = KafkaConsumer(
    "transactions_blocked",
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    auto_offset_reset="latest",
    enable_auto_commit=True,
    group_id="shap-analysis-group"
)

print("Waiting for a fraud transaction...")

# Create SHAP explainer
explainer = shap.TreeExplainer(model)

# Get one blocked transaction
for message in consumer:
    txn = message.value
    print("\nFraud transaction received:", txn)
    
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
        shap_val = shap_values[0][:,1]

    base = float(explainer.expected_value[1])

    explanation = shap.Explanation(
        values=shap_val,
        base_values=base,
        data=X.iloc[0],
        feature_names=X.columns
    )

    shap.plots.waterfall(explanation, show=False)
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    # Human readable explanation
    values = [float(v) if not hasattr(v,"__len__") else float(v[0]) for v in shap_val]

    feature_impacts = list(zip(X.columns, values))
    feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)

    print("\n🚨 FRAUD EXPLANATION")
    print("----------------------")

    reasons = []

    for feature,val in feature_impacts[:3]:
        if val > 0:
            reason = f"{feature} increased fraud risk"
        else:
            reason = f"{feature} reduced fraud risk"

        reasons.append(reason)
        print("•", reason)

    summary = "Transaction flagged as FRAUD mainly because " + ", ".join(reasons[:2])
    print("\nSummary:", summary)
