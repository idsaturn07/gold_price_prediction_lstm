from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ask_ai(prediction, sentiment, question):
    prompt = f"""
    You are a financial assistant.

    Gold price (per gram): {prediction}
    Market sentiment: {sentiment}

    User question: {question}

    Give a clear, helpful, and human-like answer.
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content