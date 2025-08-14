from dotenv import load_dotenv
import os
import requests

load_dotenv()

GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")
BASE_URL = "https://gnews.io/api/v4"

def fetch_top_headlines(topic=None, country="us", max_results=5):
    if not GNEWS_API_KEY:
        return "GNews API key is not set."

    url = f"{BASE_URL}/top-headlines"
    params = {
        "token": GNEWS_API_KEY,
        "lang": "en",
        "country": country,
        "max": max_results
    }
    if topic:
        params["topic"] = topic

    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()

        articles = data.get("articles", [])
        if not articles:
            return "I couldn't find any news articles at the moment."

        results = []
        for i, article in enumerate(articles, start=1):
            title = article.get("title", "No title")
            description = article.get("description") or ""
            content = article.get("content") or ""
            source = article.get("source", {}).get("name", "Unknown source")
            url = article.get("url", "")
            results.append(
                f"{i}. {title} ({source})\nDescription: {description}\nContent: {content}\nURL: {url}"
            )

        return "Here are the top news articles:\n\n" + "\n\n".join(results)

    except requests.exceptions.RequestException as e:
        return f"Sorry, I couldn't fetch news right now. ({e})"


def fetch_articles_by_keyword(keyword, max_results=5):
    if not GNEWS_API_KEY:
        return "GNews API key is not set."

    url = f"{BASE_URL}/search"
    params = {
        "q": keyword,
        "token": GNEWS_API_KEY,
        "lang": "en",
        "max": max_results
    }

    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()

        articles = data.get("articles", [])
        if not articles:
            return f"I couldn't find any articles about {keyword}."

        results = []
        for i, article in enumerate(articles, start=1):
            title = article.get("title", "No title")
            description = article.get("description") or ""
            content = article.get("content") or ""
            source = article.get("source", {}).get("name", "Unknown source")
            url = article.get("url", "")
            results.append(
                f"{i}. {title} ({source})\nDescription: {description}\nContent: {content}\nURL: {url}"
            )

        return f"Here are some articles about {keyword}:\n\n" + "\n\n".join(results)

    except requests.exceptions.RequestException as e:
        return f"Sorry, I couldn't fetch news right now. ({e})"
