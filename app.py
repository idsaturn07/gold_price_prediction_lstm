import streamlit as st
from model import get_gold_prediction, get_news_sentiment
from chatbot import ask_ai

st.set_page_config(page_title="Gold Price Prediction", layout="wide")

st.title("Gold Price Prediction System")

# SIDEBAR
st.sidebar.header("Settings")

currency = st.sidebar.selectbox(
    "Select Currency",
    ["USD", "INR", "EUR", "GBP", "JPY"]
)

country = st.sidebar.selectbox(
    "Select Country",
    ["Global", "India", "USA", "UK", "Japan"]
)

st.write(f"🌍 Showing data for: **{country}**")

# GET DATA
try:
    df, prediction, mae, rmse = get_gold_prediction()
except:
    st.error("Error fetching gold data. Please try again.")
    st.stop()

# CURRENCY CONVERSION
conversion_rates = {
    "USD": 1,
    "INR": 83,
    "EUR": 0.92,
    "GBP": 0.78,
    "JPY": 150
}

df_converted = df.copy()
df_converted['Close'] = df_converted['Close'] * conversion_rates[currency]

# PRICE CALCULATION
predicted_ounce_price = prediction * conversion_rates[currency]
price_per_gram = predicted_ounce_price / 28.3495

# GRAPH
st.subheader(f"Gold Price Trend ({currency})")
st.line_chart(df_converted['Close'])

# PRICE DISPLAY
st.subheader("Predicted Gold Price (Tomorrow)")

col1, col2 = st.columns(2)
col1.metric("Per Ounce", f"{predicted_ounce_price:.2f} {currency}")
col2.metric("Per Gram", f"{price_per_gram:.2f} {currency}")

st.caption("Note: Prediction is based on global gold futures (GC=F)")

# MODEL PERFORMANCE
st.subheader("Model Performance")

col1, col2 = st.columns(2)
col1.metric("MAE", f"{mae:.2f}")
col2.metric("RMSE", f"{rmse:.2f}")

# SENTIMENT
st.subheader("News Sentiment")

st.info(
    "Sentiment analysis evaluates whether news is positive, negative, or neutral. "
    "Polarity score (-1 to +1) shows intensity of sentiment."
)

articles, sentiments, polarities = get_news_sentiment(country)

# DISPLAY NEWS
for i, (title, link) in enumerate(articles):
    st.markdown(f"**[{title}]({link})**")
    st.write(f"Sentiment: {sentiments[i]}")
    st.write(f"Polarity: {polarities[i]:.3f}")

    if polarities[i] > 0:
        st.success("Positive market signal")
    elif polarities[i] < 0:
        st.error("Negative market signal")
    else:
        st.info("Neutral signal")

    st.write("---")

# OVERALL POLARITY
st.subheader("Overall Market Sentiment")

if polarities:
    avg_polarity = sum(polarities) / len(polarities)

    st.write(f"Average Polarity: {avg_polarity:.3f}")

    if avg_polarity > 0.1:
        st.success("Overall Sentiment: Positive (Market Optimistic)")
        overall_sentiment = "Positive"
    elif avg_polarity < -0.1:
        st.error("Overall Sentiment: Negative (Market Risk)")
        overall_sentiment = "Negative"
    else:
        st.info("Overall Sentiment: Neutral (Market Stable)")
        overall_sentiment = "Neutral"
else:
    overall_sentiment = "Neutral"
    st.info("No sufficient news data available")

# COUNTRY IMPACT 
st.subheader(f"{country} Market Impact")

if overall_sentiment == "Positive":
    st.success(f"In {country}, positive sentiment may increase gold demand and prices.")
elif overall_sentiment == "Negative":
    st.error(f"In {country}, negative sentiment may reduce investor confidence.")
else:
    st.info(f"In {country}, stable sentiment indicates balanced market conditions.")

# CHATBOT 
st.subheader("AI Assistant")

user_input = st.text_input("Ask about gold market")

if user_input:
    response = ask_ai(price_per_gram, overall_sentiment, user_input, country)
    st.write(response)