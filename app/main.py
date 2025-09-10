from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("fraud_model.pkl")

@app.get("/")
def home():
    return {"message": "Fraud AI Service is running"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    proba = model.predict_proba(df)[0][1]  # Probabilitas fraud
    return {"fraud_risk": float(proba)}
