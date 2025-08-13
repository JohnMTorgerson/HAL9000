from dotenv import load_dotenv
import os
import requests

load_dotenv()

APP_ID = os.getenv("WOLFRAM_ALPHA_APP_ID")
BASE_URL = "https://www.wolframalpha.com/api/v1/llm-api"

def fetch_wolfram_answer(query):
    if not APP_ID:
        return "Wolfram Alpha App ID is not set."

    url = BASE_URL
    params = {
        "appid": APP_ID,
        "input": query,
    }

    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()

        # If the API returns plain text, just return response.text
        return response.text.strip()

    except requests.exceptions.RequestException as e:
        return f"Sorry, I couldn't get an answer from Wolfram Alpha. ({e})"
