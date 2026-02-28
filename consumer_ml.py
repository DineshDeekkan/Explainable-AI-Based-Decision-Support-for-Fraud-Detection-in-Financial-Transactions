from kafka import KafkaConsumer, KafkaProducer
import json
import joblib
import pandas as pd

# Load trained model and encoders
model = joblib.load("fraud_model.pkl")
location_encoder = joblib.load("location_encoder.pkl")
device_encoder = joblib.load("device_encoder.pkl")

# Kafka Consumer
consumer = KafkaConsumer(
    "transactions",
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    auto_offset_reset="latest",
    enable_auto_commit=True,
    group_id="ml-fraud-detector"
)

# Kafka Producer
producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

print("ML-Based Fraud Detection Consumer Started...")

total = 0
correct = 0

for message in consumer:
    txn = message.value
    total += 1

    try:
        # Encode incoming transaction
        location_encoded = location_encoder.transform([txn["location"]])[0]
        device_encoded = device_encoder.transform([txn["device"]])[0]

        features = pd.DataFrame([{
            "amount": txn["amount"],
            "location": location_encoded,
            "device": device_encoded,
            "risk_score": txn["risk_score"]
        }])

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        predicted_label = "FRAUD" if prediction == 1 else "GENUINE"

        txn["fraud_probability"] = round(probability, 3)

        # Evaluate live accuracy
        if predicted_label == txn["true_label"]:
            correct += 1

        accuracy = (correct / total) * 100

        txn["predicted_label"] = predicted_label

        # Route transaction
        if predicted_label == "FRAUD":
            producer.send("transactions_blocked", txn)
        else:
            producer.send("transactions_approved", txn)

        print("Transaction:", txn)
        print(f"Fraud Probability: {probability:.3f}")
        print(f"Live ML Accuracy: {accuracy:.2f}%")
        print("-" * 60)

    except Exception as e:
        print("Error processing transaction:", e)