import yfinance as yf
import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from textblob import TextBlob
import os
from dotenv import load_dotenv
import tensorflow as tf
import random
import time

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")


def get_gold_prediction():
    df = None

    for _ in range(3):
        try:
            df = yf.download("GC=F", period="5y", progress=False)
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                break
        except:
            time.sleep(1)

    if df is None or df.empty:
        prices = np.linspace(1800, 2200, 200)
        df = pd.DataFrame({
            "Open": prices,
            "High": prices,
            "Low": prices,
            "Close": prices
        })

    df = df[['Open', 'High', 'Low', 'Close']].dropna()
    df.columns.name = None
    df.index.name = None

    print("\nDATA INFO")
    print(df.head().to_string())
    print("\nColumns:", df.columns)
    print("Rows:", len(df))
    print("Start:", df.index.min())
    print("End:", df.index.max())

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    time_step = 100

    X, y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i])
        y.append(scaled_data[i][3])

    X, y = np.array(X), np.array(y)

    if len(X) == 0:
        return df, df['Close'].values[-1], 0, 0

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        Input(shape=(X.shape[1], X.shape[2])),
        LSTM(100, return_sequences=True),
        Dropout(0.3),
        LSTM(100),
        Dropout(0.3),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import math

    y_pred = model.predict(X_test)

    y_test_reshaped = np.zeros((len(y_test), 4))
    y_pred_reshaped = np.zeros((len(y_pred), 4))

    y_test_reshaped[:, 3] = y_test
    y_pred_reshaped[:, 3] = y_pred.flatten()

    y_test_actual = scaler.inverse_transform(y_test_reshaped)[:, 3]
    y_pred_actual = scaler.inverse_transform(y_pred_reshaped)[:, 3]

    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    rmse = math.sqrt(mean_squared_error(y_test_actual, y_pred_actual))

    print("\nMODEL PERFORMANCE")
    print("MAE:", mae)
    print("RMSE:", rmse)

    last_seq = scaled_data[-time_step:]
    last_seq = np.reshape(last_seq, (1, time_step, 4))

    pred = model.predict(last_seq)

    temp = np.zeros((1, 4))
    temp[0][3] = pred

    predicted_price = scaler.inverse_transform(temp)[0][3]

    return df, predicted_price, mae, rmse


def get_news_sentiment(country="Global"):
    try:
        query_map = {
            "Global": "gold price",
            "India": "gold price india economy",
            "USA": "gold price usa fed inflation",
            "UK": "gold price uk economy",
            "Japan": "gold price japan yen"
        }

        query = query_map.get(country, "gold price")

        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}"
        res = requests.get(url)

        articles_data = []
        sentiments = []
        polarities = []

        if res.status_code == 200:
            articles = res.json()['articles']

            important_words = [
                "gold", "price", "market", "inflation",
                "interest", "rates", "economy", "demand", "central bank"
            ]

            for a in articles:
                title = a['title']
                desc = a['description'] or ""
                link = a['url']

                text = (title + " " + desc).lower()

                if "gold" not in text:
                    continue

                score = 0
                for word in important_words:
                    if word in text:
                        score += 1

                if score < 2:
                    continue

                polarity = TextBlob(text).sentiment.polarity

                print(f"[Score: {score}] Polarity: {polarity:.3f} | {title[:60]}")

                if polarity > 0:
                    s = "Positive"
                elif polarity < 0:
                    s = "Negative"
                else:
                    s = "Neutral"

                articles_data.append((title, link))
                sentiments.append(s)
                polarities.append(polarity)

                if len(articles_data) == 5:
                    break

        if not articles_data:
            return [], ["Neutral"], [0]

        return articles_data, sentiments, polarities

    except:
        return [], ["Neutral"], [0]


if __name__ == "__main__":
    df, pred, mae, rmse = get_gold_prediction()

    print("\nTEST OUTPUT")
    print("Predicted Price:", pred)
    print("MAE:", mae)
    print("RMSE:", rmse)

    articles, sentiments, polarities = get_news_sentiment()

    print("\nNEWS SENTIMENT")
    for i in range(len(articles)):
        print(sentiments[i], "|", polarities[i], "|", articles[i][0])