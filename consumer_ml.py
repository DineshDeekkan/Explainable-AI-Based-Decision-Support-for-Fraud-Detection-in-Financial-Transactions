from kafka import KafkaConsumer, KafkaProducer
import json
import joblib
import pandas as pd
from twilio.rest import Client
import requests

# Twilio whatsapp config
ACCOUNT_SID = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
AUTH_TOKEN = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

client = Client(ACCOUNT_SID, AUTH_TOKEN)

FROM_NUMBER = "whatsapp:+14xxxxxxxxx"   
TO_NUMBER = "whatsapp:+91xxxxxxxxxx"    

def send_whatsapp_alert(message):
    try:
        client.messages.create(
            body=message,
            from_=FROM_NUMBER,
            to=TO_NUMBER
        )
        print("📲 WhatsApp Alert Sent")
    except Exception as e:
        print("WhatsApp Error:", e)

# Alert message generation
def generate_alert_message(txn, reason):
    return f"""
*Transaction Alert*

A transaction of ₹{txn['amount']} from {txn['location']} was blocked.

*Reason:* {reason}

If this was you, please try again or contact support.

Ref: {txn['txn_id']}

Stay safe,
Bank Security Team
"""

# Reason generation
def get_reason(txn):
    reasons = []

    if txn["amount"] > 15000:
        reasons.append("high transaction amount")

    if txn["location"] == "Foreign":
        reasons.append("unusual location")

    if txn["risk_score"] > 1:
        reasons.append("suspicious activity pattern")

    return ", ".join(reasons) if reasons else "unusual transaction behavior"

# Load model & encoders
model = joblib.load("fraud_model.pkl")
location_encoder = joblib.load("location_encoder.pkl")
device_encoder = joblib.load("device_encoder.pkl")

consumer = KafkaConsumer(
    "transactions",
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    auto_offset_reset="latest",
    enable_auto_commit=True,
    group_id="ml-fraud-detector"
)

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

print("ML-Based Fraud Detection Engine Started...")

# Metrics
total = 0
correct = 0
TP, TN, FP, FN = 0, 0, 0, 0


def safe_encode(encoder, value):
    try:
        return encoder.transform([value])[0]
    except:
        return -1


for message in consumer:
    txn = message.value
    total += 1

    try:
        # Encoding
        location_encoded = safe_encode(location_encoder, txn["location"])
        device_encoded = safe_encode(device_encoder, txn["device"])

        features = pd.DataFrame([{
            "amount": txn["amount"],
            "location": location_encoded,
            "device": device_encoded,
            "risk_score": txn["risk_score"]
        }])

        probability = model.predict_proba(features)[0][1]

        # Dynamic threshold
        if txn["risk_score"] > 1.5:
            threshold = 0.40
        elif txn["risk_score"] > 0.8:
            threshold = 0.50
        elif txn["risk_score"] > 0.5:
            threshold = 0.60
        else:
            threshold = 0.75

        predicted_label = "FRAUD" if probability > threshold else "GENUINE"

        # Rule overrides
        if txn["location"] == "Foreign" and txn["amount"] > 15000:
            predicted_label = "FRAUD"

        if txn["risk_score"] > 2 or probability > 0.85:
            predicted_label = "FRAUD"

        if probability < 0.85 and txn["risk_score"] < 1:
            predicted_label = "GENUINE"

        txn["fraud_probability"] = round(float(probability), 3)
        txn["predicted_label"] = predicted_label

        true_label = txn["true_label"]

        # Accuracy
        if predicted_label == true_label:
            correct += 1

        accuracy = correct / total

        # Confusion Matrix
        if true_label == "FRAUD" and predicted_label == "FRAUD":
            TP += 1
        elif true_label == "GENUINE" and predicted_label == "GENUINE":
            TN += 1
        elif true_label == "GENUINE" and predicted_label == "FRAUD":
            FP += 1
        elif true_label == "FRAUD" and predicted_label == "GENUINE":
            FN += 1

        # Precision & Recall
        precision = TP / (TP + FP) if (TP + FP) else 0
        recall = TP / (TP + FN) if (TP + FN) else 0

        # Send metrics
        producer.send("model_metrics", {
            "accuracy": round(accuracy, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "TP": TP, "TN": TN, "FP": FP, "FN": FN
        })

        # Routing to kafka topics
        if predicted_label == "FRAUD":
            producer.send("transactions_blocked", txn)
            # 🔥 Send alert only for high-risk fraud
            if probability > 0.30:
                reason = get_reason(txn)
                message = generate_alert_message(txn, reason)

                send_whatsapp_alert(message)
                print("---Whatsapp alert sent---")
        else:
            producer.send("transactions_approved", txn)

        # Logs 
        print("Transaction:", txn)
        print(f"Fraud Probability: {probability:.3f}")
        print(f"Live Accuracy: {accuracy:.2f}%")
        print(f"Precision: {precision:.2f} | Recall: {recall:.2f}")
        print(f"TP:{TP} | TN:{TN} | FP:{FP} | FN:{FN}")
        print("-" * 60)

        # Reset every 100 transactions (optional)
        #if total % 100 == 0:
            #print("----- RESETTING METRICS -----")
            #total, correct = 0, 0
            #TP, TN, FP, FN = 0, 0, 0, 0

    except Exception as e:
        print("Error processing transaction :", e)
