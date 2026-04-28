from kafka import KafkaConsumer
import json
import psycopg2

# =============================
# DB CONNECTION
# =============================
conn = psycopg2.connect(
    host="localhost",
    database="fy project",
    user="postgres",
    password="Dinesh"
)
cursor = conn.cursor()

# =============================
# KAFKA CONSUMER
# =============================
consumer = KafkaConsumer(
    "transactions",
    "transactions_approved",
    "transactions_blocked",
    "model_metrics",
    "shap_insights",
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    auto_offset_reset="latest",
    group_id="kafka-postgres-group"
)

print("🚀 Kafka → PostgreSQL consumer started...")

# =============================
# INSERT FUNCTIONS
# =============================

def insert_all_transactions(data):
    query = """
    INSERT INTO transactions
    (txn_id, user_id, amount, location, device, timestamp, risk_score, true_label)
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
    ON CONFLICT (txn_id) DO NOTHING;
    """
    cursor.execute(query, (
        data["txn_id"], data["user_id"], data["amount"],
        data["location"], data["device"], data["timestamp"],
        data["risk_score"], data["true_label"]
    ))

def insert_approved(data):
    query = """
    INSERT INTO transactions_approved
    (txn_id, user_id, amount, location, device, timestamp,
     risk_score, true_label, predicted_label, fraud_probability)
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    ON CONFLICT (txn_id) DO NOTHING;
    """
    cursor.execute(query, (
        data["txn_id"], data["user_id"], data["amount"],
        data["location"], data["device"], data["timestamp"],
        data["risk_score"], data["true_label"],
        data["predicted_label"], data["fraud_probability"]
    ))

def insert_blocked(data):
    query = """
    INSERT INTO transactions_blocked
    (txn_id, user_id, amount, location, device, timestamp,
     risk_score, true_label, predicted_label, fraud_probability)
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    ON CONFLICT (txn_id) DO NOTHING;
    """
    cursor.execute(query, (
        data["txn_id"], data["user_id"], data["amount"],
        data["location"], data["device"], data["timestamp"],
        data["risk_score"], data["true_label"],
        data["predicted_label"], data["fraud_probability"]
    ))

def insert_metrics(data):
    query = """
    INSERT INTO model_metrics
    (accuracy, precision, recall, tp, tn, fp, fn)
    VALUES (%s,%s,%s,%s,%s,%s,%s);
    """
    cursor.execute(query, (
        data["accuracy"], data["precision"], data["recall"],
        data["TP"], data["TN"], data["FP"], data["FN"]
    ))

def insert_shap(data):
    txn_id = data.get("txn_id")

    features = data.get("features", [])
    impacts = data.get("impacts", [])
    summary = data.get("summary", "")

    for i, (f, imp) in enumerate(zip(features, impacts)):
        query = """
        INSERT INTO shap_insights
        (txn_id, feature_name, impact, rank, summary)
        VALUES (%s,%s,%s,%s,%s);
        """
        cursor.execute(query, (txn_id, f, imp, i+1, summary))

# =============================
# STREAM PROCESSING LOOP
# =============================

for message in consumer:
    topic = message.topic
    data = message.value

    try:
        if topic == "transactions":
            insert_all_transactions(data)

        elif topic == "transactions_approved":
            insert_approved(data)

        elif topic == "transactions_blocked":
            insert_blocked(data)

        elif topic == "model_metrics":
            insert_metrics(data)

        elif topic == "shap_insights":
            insert_shap(data)

        conn.commit()

        print(f"✅ Inserted into {topic}")

    except Exception as e:
        print("❌ Error:", e)
        conn.rollback()