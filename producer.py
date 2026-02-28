from kafka import KafkaProducer
import json
import time
import random
from collections import defaultdict

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

locations = ["Delhi", "Mumbai", "Chennai", "Kolkata", "Bangalore", "Hyderabad", "Foreign"]
devices = ["mobile", "web"]

last_txn_time = defaultdict(float)

print("Starting live transaction producer...")

while True:
    user_id = f"U{random.randint(100, 999)}"
    device = random.choice(devices)
    current_time = time.time()

    # 🔥 Base fraud probability (around 8–12%)
    fraud_probability = 0.08
    fraud_flag = False

    # -------- Generate realistic transaction --------
    amount = random.randint(100, 20000)
    location = random.choice(locations)

    risk_score = 0

    # Rule 1: High amount increases risk (but not always fraud)
    if amount > 15000:
        risk_score += random.uniform(0.5, 1.5)
        if random.random() < 0.4:
            fraud_flag = True

    # Rule 2: Foreign location increases risk (but some genuine allowed)
    if location == "Foreign":
        risk_score += random.uniform(0.5, 1.5)
        if random.random() < 0.5:
            fraud_flag = True

    # Rule 3: Rapid transactions increase risk
    if current_time - last_txn_time[user_id] < 5:
        risk_score += random.uniform(0.5, 1.5)
        if random.random() < 0.6:
            fraud_flag = True

    last_txn_time[user_id] = current_time

    # Add some random fraud noise
    if not fraud_flag and random.random() < fraud_probability:
        fraud_flag = True

    risk_score = round(risk_score, 2)

    # Final label
    true_label = "FRAUD" if fraud_flag else "GENUINE"

    transaction = {
        "txn_id": f"txn_{int(time.time()*1000)}_{random.randint(100,999)}",
        "user_id": user_id,
        "amount": amount,
        "location": location,
        "device": device,
        "timestamp": current_time,
        "risk_score": risk_score,
        "true_label": true_label
    }

    producer.send("transactions", transaction)
    print("Transaction sent:", transaction)

    time.sleep(2)