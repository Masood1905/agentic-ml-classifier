# Agentic Multi-Model Classification System

An AI agent that lets you run machine learning models by typing a request in plain English.
A large language model (LLM) reads your request, decides which of three ML models to use, extracts the input values,
**runs the model**, and explains the prediction back to you.

Built with **LangChain**, **Groq** (LLM), **scikit-learn** and **Streamlit** as a team project at COMSATS University Islamabad.

## How it works

```
User prompt ──► LLM manager (Groq, via LangChain tool calling)
                    │  picks a tool and extracts the inputs
                    ▼
     ┌──────────────┼───────────────────┐
     ▼              ▼                   ▼
 Credit card     AAPL next-day       E-commerce
 fraud (SVM)     direction (RF)      returns (LogReg)
     └──────────────┼───────────────────┘
                    ▼
     tool result ──► LLM explains it in plain English
```

The app shows a **reasoning trace** for every answer: which tool was called, with which inputs, and what it returned.
The LLM is instructed never to invent a prediction. Every prediction comes from an ML model.

## The three ML models

| Tool | Model | Data | Result on held-out test data |
|---|---|---|---|
| Credit card fraud detection | Support Vector Machine (RBF) | [Kaggle Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), 284,807 transactions (0.17% fraud) | **Caught 87 of 98 frauds (88.8% recall)**, ROC-AUC 0.975, while flagging 684 of 56,864 legit transactions (1.2%) |
| AAPL next-day direction | Random Forest | Apple daily prices, Feb 2014 – Feb 2019 (1,258 days) | 47.6% accuracy vs 53.2% for always guessing "Up", so **no better than chance** |
| E-commerce return prediction | Logistic Regression | 172,838 website visits (13.1% returned) | ROC-AUC 0.554, so **only slightly better than chance** |

## What I learned (and fixed)

The first version of this project reported near-perfect accuracy. Investigating that taught me the most important lessons:

- **Data leakage in the stock model.** The target ("did the price go up?") was calculated from the `change` column, which was also an input. The model was reading the answer. I changed the target to *tomorrow's* direction and used a time-based train/test split, so the model never sees the future.
- **Data leakage in the e-commerce model.** `returned_amount` is above 0 for exactly the returned purchases, and `sales` is 0 for every returned purchase. Either column alone gives ~100% accuracy. Without them, the remaining visit details carry very little information about returns.
- **Realistic evaluation for fraud.** I hold out a test set with the real fraud rate (0.17%) *before* undersampling the training data, so the results reflect real-world performance, not a balanced toy test set.
- **Honest results are more useful than impressive ones.** Next-day stock direction is close to random, which matches what I found with ARIMA on NVIDIA stock. The agent repeats this caution in its answers.

## Run it yourself

1. Install the requirements (Python 3.10+):
   ```
   pip install -r requirements.txt
   ```
2. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and put it in the `data/` folder. (It's 150 MB, too large for GitHub.)
3. Train the three models (takes under a minute):
   ```
   python train_aapl.py
   python train_ecommerce.py
   python train_creditcard.py
   ```
4. Get a free API key from [console.groq.com/keys](https://console.groq.com/keys). Copy `.env.example`, rename the copy to `.env`, and paste the key after `GROQ_API_KEY=`.
5. Start the app:
   ```
   streamlit run app.py
   ```
   Use the example buttons in the sidebar to try real fraud and legit transactions from the test set.

## Project structure

```
app.py                  Streamlit chat interface
agent.py                LLM manager: tool-calling loop (LangChain + Groq)
ml_tools.py             The three ML models wrapped as LangChain tools
train_aapl.py           Trains the AAPL next-day direction model (Random Forest)
train_ecommerce.py      Trains the e-commerce return model (Logistic Regression)
train_creditcard.py     Trains the credit card fraud model (SVM)
data/                   Datasets (creditcard.csv downloaded separately)
models/                 Trained models (.pkl), created by the training scripts
```

## Tools

Python · LangChain · Groq · scikit-learn · Pandas · Streamlit

*Educational project. The stock prediction is not investment advice.*
