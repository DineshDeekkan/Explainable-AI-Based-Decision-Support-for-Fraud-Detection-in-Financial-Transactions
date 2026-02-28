from kafka import KafkaConsumer, KafkaProducer
import json
import csv
import os
from collections import defaultdict

# ==============================
# Kafka Consumer
# ==============================
consumer = KafkaConsumer(
    "transactions",
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    auto_offset_reset="latest",
    enable_auto_commit=True,
    group_id="fraud-detector-group"
)

# ==============================
# Kafka Producer (routing)
# ==============================
producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

# ==============================
# CSV File Setup
# ==============================
file_path = "transactions_dataset.csv"

file_exists = os.path.isfile(file_path)

csv_file = open(file_path, mode="a", newline="")
csv_writer = csv.writer(csv_file)

# Write header only once
if not file_exists:
    csv_writer.writerow([
        "txn_id",
        "user_id",
        "amount",
        "location",
        "device",
        "timestamp",
        "risk_score",
        "true_label",
        "predicted_label"
    ])

# ==============================
# Rapid Transaction Tracker
# ==============================
last_txn_time = defaultdict(float)

# Evaluation Metrics
total = 0
correct = 0

print("Fraud Detection Consumer Started...")

for message in consumer:
    txn = message.value
    total += 1

    user_id = txn["user_id"]
    amount = txn["amount"]
    location = txn["location"]
    device = txn["device"]
    timestamp = txn["timestamp"]
    true_label = txn["true_label"]

    fraud = False

    # Rule 1
    if amount > 15000:
        fraud = True

    # Rule 2
    if location == "Foreign" and amount > 10000:
        fraud = True

    # Rule 3
    if timestamp - last_txn_time[user_id] < 5:
        fraud = True

    # Rule 4
    if txn["risk_score"] >= 2.5:
        fraud = True

    last_txn_time[user_id] = timestamp

    predicted_label = "FRAUD" if fraud else "GENUINE"

    # Accuracy calculation
    if predicted_label == true_label:
        correct += 1

    accuracy = (correct / total) * 100

    txn["predicted_label"] = predicted_label

    # Save to CSV
    csv_writer.writerow([
        txn["txn_id"],
        user_id,
        amount,
        location,
        device,
        timestamp,
        txn.get("risk_score", 0), # To be checked
        true_label,
        predicted_label
    ])

    csv_file.flush()

    # Route to correct topic
    if predicted_label == "FRAUD":
        producer.send("transactions_blocked", txn)
    else:
        producer.send("transactions_approved", txn)

    print("Transaction:", txn)
    print(f"Live Accuracy: {accuracy:.2f}%")
    print("-" * 60)