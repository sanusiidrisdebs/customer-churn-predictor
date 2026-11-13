import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# 1. Load public Telco customer churn dataset
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
print("Downloading and preparing dataset...")
df = pd.read_csv(url)

# 2. Select 4 high-impact features for a fast baseline
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

features = ["tenure", "MonthlyCharges", "TotalCharges"]
X = df[features]
y = df["Churn"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Model
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
print("\nModel Trained Successfully! Performance:")
print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))

# 6. Save artifact
joblib.dump(model, "churn_model.pkl")
print("Saved model to 'churn_model.pkl'")