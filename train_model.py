import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, classification_report, confusion_matrix


# ==============================
# 1️⃣ Load Dataset
# ==============================

df = pd.read_csv("transactions_dataset.csv")

print("Dataset Loaded Successfully")
print(df.head())

# ==============================
# 2️⃣ Feature Engineering
# ==============================

# Encode categorical features
location_encoder = LabelEncoder()
device_encoder = LabelEncoder()

df["location"] = location_encoder.fit_transform(df["location"])
df["device"] = device_encoder.fit_transform(df["device"])

# Encode target
df["true_label"] = df["true_label"].map({"GENUINE": 0, "FRAUD": 1})

# Features and Target
X = df[["amount", "location", "device", "risk_score"]]
y = df["true_label"]

# ==============================
# 3️⃣ Train-Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y,    # 🔥 Keeps fraud ratio same in train & test
    random_state=42
)

print("\nClass Distribution:")
print(y.value_counts())

# ==============================
# 4️⃣ Train RandomForest (With Class Weight)
# ==============================

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=7,
    class_weight='balanced',   # 🔥 Important for imbalance
    random_state=42
)

model.fit(X_train, y_train)

# ==============================
# 5️⃣ Evaluate Model
# ==============================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy * 100, "%")
print("\nFraud Recall:", recall_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ==============================
# 6️⃣ Save Model + Encoders
# ==============================

joblib.dump(model, "fraud_model.pkl")
joblib.dump(location_encoder, "location_encoder.pkl")
joblib.dump(device_encoder, "device_encoder.pkl")

print("\nModel and Encoders Saved Successfully ✅")