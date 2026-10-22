# train_creditcard.py
# Detects fraudulent credit card transactions with a Support Vector Machine (SVM).
# Dataset: Kaggle "Credit Card Fraud Detection" (creditcard.csv) -> put it in the data/ folder.
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import resample

BASE = Path(__file__).resolve().parent
(BASE / "models").mkdir(exist_ok=True)

cc = pd.read_csv(BASE / "data" / "creditcard.csv")  # columns: Time, V1..V28, Amount, Class
FEATURES = [c for c in cc.columns if c != "Class"]

# 1) Hold out a REALISTIC test set first (fraud is only ~0.17% of transactions),
#    so the results reflect how the model would perform on real traffic.
train, test = train_test_split(cc, test_size=0.2, random_state=42, stratify=cc["Class"])

# 2) Undersample legit transactions in the TRAINING set only (3 legit : 1 fraud),
#    because SVMs are slow on 200k+ rows and the classes are extremely imbalanced.
fraud = train[train["Class"] == 1]
legit = resample(train[train["Class"] == 0], replace=False, n_samples=len(fraud) * 3, random_state=42)
train_bal = pd.concat([legit, fraud])

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(train_bal[FEATURES])
X_test_sc = scaler.transform(test[FEATURES])

clf = SVC(kernel="rbf", probability=True, C=1.0, random_state=42)
clf.fit(X_train_sc, train_bal["Class"])

preds = clf.predict(X_test_sc)
auc = roc_auc_score(test["Class"], clf.predict_proba(X_test_sc)[:, 1])
tn, fp, fn, tp = confusion_matrix(test["Class"], preds).ravel()

print("=== Credit Card Fraud Detector (SVM) ===")
print(f"Training rows (balanced): {len(train_bal):,}  |  Test rows (realistic): {len(test):,}")
print(f"ROC-AUC: {auc:.3f}")
print(f"Frauds caught: {tp} of {tp + fn} ({tp / (tp + fn):.1%})  |  Legit transactions wrongly flagged: {fp:,} of {fp + tn:,}")
print(classification_report(test["Class"], preds, target_names=["Legit", "Fraud"], digits=3, zero_division=0))

joblib.dump({"model": clf, "scaler": scaler, "features": FEATURES}, BASE / "models" / "creditcard_model.pkl")
print("Saved models/creditcard_model.pkl")
