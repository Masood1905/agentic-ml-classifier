# ml_tools.py
# The three ML "workers". Each one is a LangChain tool that the LLM agent can call.
from pathlib import Path

import joblib
import pandas as pd
from langchain_core.tools import tool

MODELS = Path(__file__).resolve().parent / "models"

for name in ("aapl_model.pkl", "ecommerce_model.pkl", "creditcard_model.pkl"):
    if not (MODELS / name).exists():
        raise FileNotFoundError(
            f"models/{name} not found. Run the training scripts first: "
            "python train_aapl.py, python train_ecommerce.py, python train_creditcard.py"
        )

aapl_bundle = joblib.load(MODELS / "aapl_model.pkl")
ecommerce_bundle = joblib.load(MODELS / "ecommerce_model.pkl")
creditcard_bundle = joblib.load(MODELS / "creditcard_model.pkl")


def _infer(bundle, values: list, labels: list) -> str:
    """Scale the inputs exactly like during training, then predict."""
    X = pd.DataFrame([values], columns=bundle["features"])
    X_sc = bundle["scaler"].transform(X)
    pred = int(bundle["model"].predict(X_sc)[0])
    proba = bundle["model"].predict_proba(X_sc)[0][pred]
    return f"Prediction: {labels[pred]} (model confidence: {proba:.1%})"


def _encode(bundle, column: str, value: str) -> int:
    """Turn a text value (e.g. 'Credit Card') into the number the model was trained on."""
    encoder = bundle["encoders"][column]
    known = {c.lower(): c for c in encoder.classes_}
    match = known.get(str(value).strip().lower())
    if match is None:
        raise ValueError(f"'{value}' is not a known value for {column}. Allowed: {', '.join(encoder.classes_)}")
    return int(encoder.transform([match])[0])


@tool
def classify_stock_direction(
    open: float, high: float, low: float, close: float, volume: float,
    change: float, changePercent: float, vwap: float, changeOverTime: float,
) -> str:
    """Predict whether Apple (AAPL) stock will close HIGHER (Up) or LOWER (Down) on the NEXT trading day,
    using today's trading values: open, high, low and close price, volume, change (today's price change
    in USD), changePercent (today's change in %), vwap (volume-weighted average price) and
    changeOverTime (cumulative change since 21 Feb 2014 as a fraction, e.g. 1.5 = +150%).
    Note: on unseen data this model is about as accurate as a coin flip."""
    values = [open, high, low, close, volume, change, changePercent, vwap, changeOverTime]
    return _infer(aapl_bundle, values, ["Down", "Up"]) + (
        " | Caution: next-day stock direction is close to random; this model performed no better than guessing on test data."
    )


@tool
def classify_ecommerce_return(
    duration_secs: float, bytes: float, network_protocol: str, accessed_from: str,
    gender: str, membership: str, language: str, pay_method: str, age: float | None = None,
) -> str:
    """Predict whether an e-commerce purchase will be Returned or Not Returned, from the website visit
    and customer details: duration_secs (visit length in seconds), bytes (data transferred),
    network_protocol (TCP, HTTP, UDP or ICMP), accessed_from (Chrome, Safari, Mozilla Firefox,
    Microsoft Edge, Android App, IOS App or Others), gender (Male, Female or Unknown),
    membership (Normal, Premium or Not Logged In), language (e.g. English, Spanish),
    pay_method (Cash, Credit Card, Debit Card or Others) and optionally the customer's age."""
    b = ecommerce_bundle
    values = [
        duration_secs, bytes, b["age_median"] if age is None else age,
        _encode(b, "network_protocol", network_protocol),
        _encode(b, "accessed_Ffom", accessed_from),
        _encode(b, "gender", gender),
        _encode(b, "membership", membership),
        _encode(b, "language", language),
        _encode(b, "pay_method", pay_method),
    ]
    return _infer(b, values, ["Not Returned", "Returned"]) + (
        " | Caution: these visit details carry very little signal about returns (ROC-AUC about 0.55), so treat this as a weak estimate."
    )


@tool
def classify_credit_card_fraud(features: list[float]) -> str:
    """Detect whether a credit card transaction is Legit or Fraud.
    Input must be a list of exactly 30 numbers in this order: [Time, V1, V2, ..., V28, Amount],
    as in the Kaggle credit card fraud dataset (V1-V28 are anonymised PCA features)."""
    if len(features) != 30:
        return f"Error: expected 30 numbers [Time, V1..V28, Amount] but received {len(features)}. Ask the user for the full list."
    return _infer(creditcard_bundle, features, ["Legit", "Fraud"])


ALL_TOOLS = [classify_stock_direction, classify_ecommerce_return, classify_credit_card_fraud]
