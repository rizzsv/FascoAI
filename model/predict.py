import pandas as pd
import joblib


model = joblib.load("fraud_model.pkl")


order = {
    "total_amount": 6000000,
    "shipping_cost": 0  # misalnya tidak ada ongkir
}

# ubah ke DataFrame
X_new = pd.DataFrame([order])

# prediksi
pred = model.predict(X_new)
print("Fraud Prediction:", "FRAUD" if pred[0] == 1 else "NORMAL")
