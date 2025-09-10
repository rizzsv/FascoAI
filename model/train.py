import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

data = {
    "total": [500000, 7000000, 150000, 9000000, 200000],
    "account_age_days": [30, 1, 45, 2, 60],
    "payment_method": [1, 0, 1, 0, 1],  # 1 = non-COD, 0 = COD
    "orders_today": [1, 6, 1, 3, 2],
    "address_match": [1, 0, 1, 0, 1],
    "fraud": [0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)
X = df.drop("fraud", axis=1)
y = df["fraud"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Model Accuracy:", model.score(X_test, y_test))

# Simpan model
joblib.dump(model, "fraud_model.pkl")
