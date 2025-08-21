import os
import requests
import urllib.parse
from dotenv import load_dotenv
from bs4 import BeautifulSoup  # pip install beautifulsoup4
import re

load_dotenv()

CORE_API_BASE = "https://api.wikimedia.org/core/v1/wikipedia/en"
REST_API_BASE = "https://en.wikipedia.org/api/rest_v1/page"
WIKI_ACCESS_TOKEN = os.getenv("WIKI_ACCESS_TOKEN")

STOP_SECTIONS = [
    "See also",
    "References",
    "Notes",
    "Citations",
    "Works cited",
    "Further reading",
    "External links"
]

def clean_html_preserve_structure(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Remove unwanted tags
    for tag in soup(["sup", "table", "style", "script", "nav"]):
        tag.decompose()

    # Add newlines before and after headings
    for level in range(1, 7):
        for tag in soup.find_all(f"h{level}"):
            heading_text = tag.get_text(strip=True)
            if heading_text in STOP_SECTIONS:
                # Remove this heading and all following siblings
                for sibling in list(tag.next_siblings):
                    if hasattr(sibling, 'decompose'):
                        sibling.decompose()
                tag.decompose()
            else:
                tag.insert_before("\n\n=== ")
                tag.insert_after(" ===\n\n")

    # Add newline after paragraphs and list items
    for tag in soup.find_all(["p", "li"]):
        tag.insert_after("\n\n")

    # Extract all text, including from divs and sections
    text = "".join(s for s in soup.strings)

    # Collapse more than 2 consecutive newlines to exactly 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing spaces from each line but keep empty lines
    text = "\n".join(line.strip() for line in text.splitlines())

    return text


def search_wikipedia(query: str, limit: int = 5):
    """Search Wikipedia using the Core API and return a list of candidate pages."""
    headers = {
        "User-Agent": f"HAL9000/1.0 ({os.getenv('WIKI_EMAIL')})"
    }
    if WIKI_ACCESS_TOKEN:
        headers["Authorization"] = f"Bearer {WIKI_ACCESS_TOKEN}"

    url = f"{CORE_API_BASE}/search/page"
    params = {
        "q": query,
        "limit": limit
    }

    resp = requests.get(url, headers=headers, params=params)
    resp.raise_for_status()
    data = resp.json()

    results = []
    for page in data.get("pages", []):
        excerpt_html = page.get("excerpt", "")
        # Strip all HTML tags from excerpt
        soup = BeautifulSoup(excerpt_html, "html.parser")
        clean_excerpt = soup.get_text(separator=" ", strip=True)
        results.append({
            "title": page["title"],
            "excerpt": clean_excerpt
        })

    return results


def fetch_wikipedia(title: str, mode: str = "summary"):
    headers = {
        "User-Agent": f"HAL9000/1.0 ({os.getenv('WIKI_EMAIL')})"
    }
    if WIKI_ACCESS_TOKEN:
        headers["Authorization"] = f"Bearer {WIKI_ACCESS_TOKEN}"

    if mode == "full":
        safe_title = urllib.parse.quote(title)

        # First try Core API mobile-html
        url = f"{CORE_API_BASE}/page/mobile-html/{safe_title}"
        resp = requests.get(url, headers=headers)

        if resp.status_code == 200:
            html = resp.text
        else:
            # Fallback: REST API html
            fallback_url = f"{REST_API_BASE}/html/{urllib.parse.quote(title.replace(' ', '_'))}"
            resp = requests.get(fallback_url, headers=headers)
            resp.raise_for_status()
            html = resp.text

        text = clean_html_preserve_structure(html)
        return {
            "mode": "full",
            "title": title,
            "text": text
        }

    else:
        # Summary always uses REST API
        safe_title = urllib.parse.quote(title.replace(" ", "_"))
        url = f"{REST_API_BASE}/summary/{safe_title}"
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return {
            "mode": "summary",
            "title": data.get("title", title),
            "extract": data.get("extract"),
            "url": data.get("content_urls", {}).get("desktop", {}).get("page")
        }


# Example usage
if __name__ == "__main__":
    query = "Alan Turing"
    print("Search results:")
    candidates = search_wikipedia(query)
    for idx, page in enumerate(candidates, 1):
        # print(f"{idx}. {page['title']}: {page.get('excerpt', '')}")
        print(page)

    if candidates:
        print("\nFetching full article for first candidate:")
        print(fetch_wikipedia(candidates[0]["title"], "full")["text"])
