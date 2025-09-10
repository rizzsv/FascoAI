import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# load data
df = pd.read_csv("data/orders.csv")

df["fraud"] = df.apply(
    lambda row: 1 if (row["total_amount"] >= 5500000 or pd.isna(row["shipping_cost"])) else 0,
    axis=1
)

X = df[["total_amount", "shipping_cost"]].fillna(0)
y = df["fraud"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump(model, "fraud_model.pkl")

print("Model trained & saved successfully!")
