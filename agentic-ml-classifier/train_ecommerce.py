# train_ecommerce.py
# Predicts whether an e-commerce purchase will be returned, from the visit and customer details.
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

BASE = Path(__file__).resolve().parent
(BASE / "models").mkdir(exist_ok=True)

data_file = next((BASE / "data").glob("E-commerce*Logs.csv"))  # works with spaces or underscores
ecom = pd.read_csv(data_file, low_memory=False)

# Target: returned (Yes = 1, No = 0)
ecom = ecom.dropna(subset=["returned"])
ecom["target"] = (ecom["returned"].str.strip().str.lower() == "yes").astype(int)

NUMERIC = ["duration_(secs)", "bytes", "age"]
# NOTE: two columns are deliberately NOT used, because they leak the answer:
#   - 'returned_amount' is above 0 exactly when the item was returned
#   - 'sales' is 0 for every returned purchase (the refund is already subtracted)
# A model using either of them scores ~100% but has learned nothing useful.
CATEGORICAL = ["network_protocol", "accessed_Ffom", "gender", "membership", "language", "pay_method"]

for col in NUMERIC:
    ecom[col] = pd.to_numeric(ecom[col], errors="coerce")  # '--' becomes missing
# About half the rows have no age: fill with the median instead of throwing those rows away
AGE_MEDIAN = float(ecom["age"].median())
ecom["age"] = ecom["age"].fillna(AGE_MEDIAN)
ecom = ecom.dropna(subset=NUMERIC)

for col in CATEGORICAL:
    ecom[col] = ecom[col].astype(str).str.strip()
ecom["accessed_Ffom"] = ecom["accessed_Ffom"].replace({"SafFRi": "Safari"})  # fix a typo in the data
ecom["language"] = ecom["language"].str.title()  # 'italian' and 'Italian' are the same language

encoders = {}
for col in CATEGORICAL:
    le = LabelEncoder()
    ecom[col + "_enc"] = le.fit_transform(ecom[col])
    encoders[col] = le

FEATURES = NUMERIC + [c + "_enc" for c in CATEGORICAL]
X, y = ecom[FEATURES], ecom["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# Only ~13% of purchases are returned, so balance the classes;
# otherwise the model just predicts "Not Returned" for everyone.
clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(X_train_sc, y_train)

preds = clf.predict(X_test_sc)
auc = roc_auc_score(y_test, clf.predict_proba(X_test_sc)[:, 1])

print("=== E-Commerce Return Predictor (Logistic Regression) ===")
print(f"Rows: {len(ecom):,}  |  Return rate: {y.mean():.1%}")
print(f"ROC-AUC: {auc:.3f}  (0.5 = random guessing, 1.0 = perfect)")
print(classification_report(y_test, preds, target_names=["Not Returned", "Returned"], zero_division=0))

joblib.dump(
    {"model": clf, "scaler": scaler, "features": FEATURES, "numeric": NUMERIC,
     "categorical": CATEGORICAL, "encoders": encoders, "age_median": AGE_MEDIAN},
    BASE / "models" / "ecommerce_model.pkl",
)
print("Saved models/ecommerce_model.pkl")
