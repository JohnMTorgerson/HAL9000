import os
import re
import math
import requests
from dotenv import load_dotenv
from places_types import PLACE_TYPE_MAP

load_dotenv()

API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
LAT = os.getenv("LAT")
LON = os.getenv("LON")

if not API_KEY or not LAT or not LON:
    raise ValueError("Missing required .env values: GOOGLE_MAPS_API_KEY, LAT, LON")

BASE_URL = "https://places.googleapis.com/v1"


def _clean_text(s: str) -> str:
    """Replace weird unicode spaces with normal spaces and trim."""
    if not isinstance(s, str):
        return s
    return re.sub(r'[\u2000-\u200F\u202F\u205F\u3000]', ' ', s).strip()


def _haversine(lat1, lon1, lat2, lon2):
    """Return distance in miles between two lat/lon pairs."""
    R = 3958.8  # Earth radius in miles
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))


def _format_results(data):
    results = []
    for place in data.get("places", []):
        name = _clean_text(place.get("displayName", {}).get("text", "Unknown"))
        addr = _clean_text(place.get("formattedAddress", "No address available"))
        open_now = place.get("currentOpeningHours", {}).get("openNow")
        open_hours = [_clean_text(h) for h in place.get("currentOpeningHours", {}).get("weekdayDescriptions", [])]

        loc = place.get("location")
        distance = None
        if loc:
            distance = _haversine(float(LAT), float(LON),
                                  float(loc.get("latitude", 0)),
                                  float(loc.get("longitude", 0)))

        results.append({
            "name": name,
            "address": addr,
            "open_now": open_now,
            "hours": open_hours,
            "distance_miles": round(distance, 1) if distance else None
        })
    return results

def search(query: str, radius=5000, max_results=5, fetch_count=20):
    """Search for places by type or text.
    Returns list of dicts (sorted by distance) or error string."""
    query_norm = query.strip().lower()
    place_type = PLACE_TYPE_MAP.get(query_norm)

    if place_type:
        url = f"{BASE_URL}/places:searchNearby"
        payload = {
            "includedTypes": [place_type],
            "maxResultCount": fetch_count,
            "locationRestriction": {
                "circle": {
                    "center": {"latitude": float(LAT), "longitude": float(LON)},
                    "radius": radius
                }
            }
        }
    else:
        url = f"{BASE_URL}/places:searchText"
        payload = {
            "textQuery": f"{query} near me",
            "maxResultCount": fetch_count,
            "locationBias": {
                "circle": {
                    "center": {"latitude": float(LAT), "longitude": float(LON)},
                    "radius": radius
                }
            }
        }

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": (
            "places.displayName,places.formattedAddress,"
            "places.currentOpeningHours,places.location"
        ),
    }

    try:
        resp = requests.post(url, headers=headers, json=payload)
        data = resp.json()
    except Exception as e:
        return f"Error contacting Places API: {e}"

    if "error" in data:
        return f"Places API error: {data['error'].get('message', 'Unknown error')}"

    if "places" not in data:
        return "No results found."

    # Format and sort by distance
    results = _format_results(data)
    results.sort(key=lambda r: r["distance_miles"] if r["distance_miles"] else float("inf"))

    return results[:max_results]


if __name__ == "__main__":
    from pprint import pprint

    print("🔍 Testing search...\n")
    for q in ["hardware store", "fast food", "Microcenter"]:
        print(f"\nSearch: {q}")
        results = search(q)
        if isinstance(results, str):
            print("Error:", results)
        else:
            for r in results:
                pprint(r)
