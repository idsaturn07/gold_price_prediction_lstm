# 🥇 Gold Price Prediction System

> AI-powered gold price forecasting with news sentiment analysis and an intelligent chatbot assistant.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-orange.svg)](https://tensorflow.org)

**GitHub:** [github.com/idsaturn07/gold_price_prediction_lstm](https://github.com/idsaturn07/gold_price_prediction_lstm)

---

## 📌 What It Does

- 🔮 Predicts **tomorrow's gold price** using a deep learning LSTM model
- 📈 Shows **5 years of gold price history** as an interactive chart
- 📊 Displays **model accuracy** (MAE & RMSE) on test data
- 📰 Fetches **live gold news** and scores sentiment (Positive / Negative / Neutral)
- 🌐 Supports **5 currencies** — USD, INR, EUR, GBP, JPY
- 🗺️ Filters news by **country** — Global, India, USA, UK, Japan
- 🤖 **AI chatbot** (LLaMA 3.1) answers gold market questions with live context

---

## 🗂️ Project Structure

```
gold_price_prediction_lstm/
├── app.py              # Streamlit UI — main entry point
├── model.py            # LSTM training, gold data, news sentiment
├── chatbot.py          # Groq LLaMA AI assistant
├── .env                # API keys (fill in yours)
├── .gitignore          # Excludes venv, pycache, .env from Git
├── LICENSE             # MIT License
├── requirements.txt    # Python dependencies
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/idsaturn07/gold_price_prediction_lstm.git
cd gold_price_prediction_lstm
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
python -m textblob.download_corpora
```

### 4. Add your API keys
Open `.env` and fill in your keys:
```
NEWS_API_KEY=your_newsapi_key_here
GROQ_API_KEY=your_groq_api_key_here
```

| Key | Get it free from |
|---|---|
| `NEWS_API_KEY` | [newsapi.org](https://newsapi.org) |
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) |

### 5. Run the app
```bash
streamlit run app.py
```

> ⏳ First load takes **2–4 minutes** — the LSTM model trains fresh on startup.

---

## 🧠 Model Details

| Parameter | Value |
|---|---|
| Data Source | Yahoo Finance — Gold Futures `GC=F` |
| History | 5 years (~1,260 trading days) |
| Features | Open, High, Low, Close |
| Sequence Length | 100 trading days |
| Architecture | 2× LSTM (100 units) + Dropout(0.3) + Dense(1) |
| Optimizer | Adam |
| Loss | Mean Squared Error |
| Epochs | 50 |
| Train / Test Split | 80% / 20% |

---

## 📰 Sentiment Analysis

News is fetched from **NewsAPI** using country-specific queries and filtered to only gold-relevant articles. Each headline is scored using **TextBlob**:

```
polarity > 0   →  Positive
polarity < 0   →  Negative
polarity = 0   →  Neutral
```

The average polarity across all articles determines the **overall market sentiment**, which is also passed to the AI chatbot as context.

---

## 🤖 AI Chatbot

Powered by **Groq — LLaMA 3.1 8B Instant**. Every question is enriched with live data before being sent to the model:

```
Gold price (per gram): $X.XX
Market sentiment: Positive / Negative / Neutral
User question: ...
```

This means the chatbot always responds with awareness of the **current predicted price and market mood**.

---

## ⚠️ Disclaimer

This project is for **educational purposes only** and does not constitute financial advice. Always consult a certified financial advisor before making investment decisions.
