import os
import requests
from datetime import datetime
from dotenv import load_dotenv
from sports_api_base import BaseSportsAPI  # assuming same directory

load_dotenv()

class APISportsAmericanFootballAPI(BaseSportsAPI):
    """Implementation of NFL data via API-Sports American Football API."""

    BASE_URL = "https://v1.american-football.api-sports.io"

    # Hardcoded map for NFL (API-Sports has multiple leagues: NFL, NCAA, CFL, etc. but we only support NFL for now)
    LEAGUE_IDS = {
        "NFL": 1,   # NFL league ID in API-Sports
    }

    TEAM_IDS = {
        "LAS VEGAS RAIDERS": 1,
        "JACKSONVILLE JAGUARS": 2,
        "NEW ENGLAND PATRIOTS": 3,
        "NEW YORK GIANTS": 4,
        "BALTIMORE RAVENS": 5,
        "TENNESSEE TITANS": 6,
        "DETROIT LIONS": 7,
        "ATLANTA FALCONS": 8,
        "CLEVELAND BROWNS": 9,
        "CINCINNATI BENGALS": 10,
        "ARIZONA CARDINALS": 11,
        "PHILADELPHIA EAGLES": 12,
        "NEW YORK JETS": 13,
        "SAN FRANCISCO 49ERS": 14,
        "GREEN BAY PACKERS": 15,
        "CHICAGO BEARS": 16,
        "KANSAS CITY CHIEFS": 17,
        "WASHINGTON COMMANDERS": 18,
        "CAROLINA PANTHERS": 19,
        "BUFFALO BILLS": 20,
        "INDIANAPOLIS COLTS": 21,
        "PITTSBURGH STEELERS": 22,
        "SEATTLE SEAHAWKS": 23,
        "TAMPA BAY BUCCANEERS": 24,
        "MIAMI DOLPHINS": 25,
        "HOUSTON TEXANS": 26,
        "NEW ORLEANS SAINTS": 27,
        "DENVER BRONCOS": 28,
        "DALLAS COWBOYS": 29,
        "LOS ANGELES CHARGERS": 30,
        "LOS ANGELES RAMS": 31,
        "MINNESOTA VIKINGS": 32,
        "NFC": 33,
        "AFC": 34
    }

    def __init__(self):
        self.api_key = os.getenv("APISPORTS_API_KEY")
        if not self.api_key:
            raise ValueError("Missing API-Sports key. Set APISPORTS_API_KEY in your environment.")
        self.headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": "v1.american-football.api-sports.io",
        }

    def _resolve_team_or_league(self, name: str):
        """Assumes SportsRouter has already normalized aliases."""
        key = name.strip().upper()

        # League direct match
        if key in self.LEAGUE_IDS:
            return ("league", self.LEAGUE_IDS[key])

        # Team direct match
        if key in self.TEAM_IDS:
            return ("team", self.TEAM_IDS[key])

        raise ValueError(f"Could not resolve '{name}' as a league or team.")

    def next_game(self, name: str):
        kind, identifier = self._resolve_team_or_league(name)
        params = {"season": self._current_season(), "next": 1}

        if kind == "league":
            params["league"] = identifier
        else:  # team
            params["team"] = identifier

        resp = requests.get(f"{self.BASE_URL}/games", headers=self.headers, params=params)
        resp.raise_for_status()
        return resp.json().get("response", [])

    def schedule(self, name: str):
        kind, identifier = self._resolve_team_or_league(name)
        params = {"season": self._current_season()}

        if kind == "league":
            params["league"] = identifier
        else:  # team
            params["team"] = identifier

        resp = requests.get(f"{self.BASE_URL}/games", headers=self.headers, params=params)
        resp.raise_for_status()
        return resp.json().get("response", [])

    def standings(self, league: str):
        key = league.strip().upper()
        if key not in self.LEAGUE_IDS:
            raise ValueError(f"Unsupported league '{league}'. Supported: {list(self.LEAGUE_IDS.keys())}")

        league_id = self.LEAGUE_IDS[key]
        resp = requests.get(
            f"{self.BASE_URL}/standings",
            headers=self.headers,
            params={"league": league_id, "season": self._current_season()},
        )
        resp.raise_for_status()
        return resp.json().get("response", [])

    def _current_season(self):
        year = datetime.now().year
        month = datetime.now().month
        # Include Jan-Mar as previous year's season
        if month < 4:
            return year - 1
        return year

