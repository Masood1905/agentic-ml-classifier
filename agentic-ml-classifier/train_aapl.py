# train_aapl.py
# Predicts whether Apple (AAPL) stock will close HIGHER or LOWER on the NEXT trading day,
# using only information available at today's close.
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
(BASE / "models").mkdir(exist_ok=True)

aapl = pd.read_csv(BASE / "data" / "AAPL.csv").sort_values("date").reset_index(drop=True)

FEATURES = ["open", "high", "low", "close", "volume", "change", "changePercent", "vwap", "changeOverTime"]

# Target: 1 if TOMORROW's close is higher than today's, else 0.
# (Using today's own 'change' as the target would leak the answer into the features.)
aapl["target"] = (aapl["change"].shift(-1) > 0).astype(int)
aapl = aapl.iloc[:-1]  # the last day has no "tomorrow"
aapl = aapl[FEATURES + ["target"]].dropna()

# Time-based split: train on the past, test on the most recent 20% of days.
# (A random split would let the model learn from days that come after the test days.)
split = int(len(aapl) * 0.8)
train, test = aapl.iloc[:split], aapl.iloc[split:]

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(train[FEATURES])
X_test_sc = scaler.transform(test[FEATURES])

clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=42)
clf.fit(X_train_sc, train["target"])

preds = clf.predict(X_test_sc)
baseline = max(test["target"].mean(), 1 - test["target"].mean())

print("=== AAPL Next-Day Direction (Random Forest) ===")
print(f"Train days: {len(train)}  |  Test days: {len(test)}")
print(f"Test accuracy           : {accuracy_score(test['target'], preds):.1%}")
print(f"Baseline (majority day) : {baseline:.1%}")
print(classification_report(test["target"], preds, target_names=["Down", "Up"], zero_division=0))

joblib.dump({"model": clf, "scaler": scaler, "features": FEATURES}, BASE / "models" / "aapl_model.pkl")
print("Saved models/aapl_model.pkl")
