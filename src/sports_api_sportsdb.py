import requests
from datetime import datetime
from sports_api_base import BaseSportsAPI  # assuming same directory

class TheSportsDBAPI(BaseSportsAPI):
    """Implementation of sports API using TheSportsDB."""

    BASE_URL = "https://www.thesportsdb.com/api/v1/json/123"

    # Hardcoded map of supported leagues and their IDs in TheSportsDB
    LEAGUE_IDS = {
        "NFL": 4391,
        "NBA": 4387,
        "MLB": 4424,
        "NHL": 4380,
        "MLS": 4346,
        "EPL": 4328,          # English Premier League
        "LALIGA": 4335,
        "SERIEA": 4332,
        "BUNDESLIGA": 4331,
        "LIGUE1": 4334,
        "UCL": 4480,          # UEFA Champions League
        "CFL": 4399,
        "NCAAF": 4393,
        "NCAAB": 4396,
        "WNBA": 4422,
    }

    def _resolve_team_or_league(self, name: str):
        """Return ('league', id) or ('team', id) depending on the parameter."""
        key = name.strip().upper()

        # Try league first
        if key in self.LEAGUE_IDS:
            return ("league", self.LEAGUE_IDS[key])

        # Otherwise treat as team: searchteams.php?t=<name>
        resp = requests.get(f"{self.BASE_URL}/searchteams.php", params={"t": name})
        data = resp.json()
        if data and data.get("teams"):
            return ("team", data["teams"][0]["idTeam"])

        raise ValueError(f"Could not resolve '{name}' as a league or team.")

    def next_game(self, name: str):
        kind, identifier = self._resolve_team_or_league(name)

        if kind == "league":
            url = f"{self.BASE_URL}/eventsnextleague.php?id={identifier}"
            resp = requests.get(url).json()
            return resp.get("events", [])
        else:  # team
            url = f"{self.BASE_URL}/eventsnext.php?id={identifier}"
            resp = requests.get(url).json()
            return resp.get("events", [])

    def schedule(self, name: str):
        kind, identifier = self._resolve_team_or_league(name)

        if kind == "league":
            url = f"{self.BASE_URL}/eventsnextleague.php?id={identifier}"
            resp = requests.get(url).json()
            return resp.get("events", [])
        else:
            url = f"{self.BASE_URL}/eventsnext.php?id={identifier}"
            resp = requests.get(url).json()
            return resp.get("events", [])

    def standings(self, league: str):
        key = league.strip().upper()
        if key not in self.LEAGUE_IDS:
            raise ValueError(f"Unsupported league '{league}'. Supported: {list(self.LEAGUE_IDS.keys())}")

        league_id = self.LEAGUE_IDS[key]
        season = self._current_season_string()
        url = f"{self.BASE_URL}/lookuptable.php?l={league_id}&s={season}"
        resp = requests.get(url)

        if not resp.text.strip():  # Empty body
            return {"error": f"Standings are not available for {league} in TheSportsDB free API."}

        try:
            data = resp.json()
        except ValueError:
            return {"error": f"Could not parse standings response for {league}."}

        return data.get("table", []) or {"error": f"No standings data found for {league}."}


    def _current_season_string(self):
        """Return a season string like '2025-2026' for current year."""
        now = datetime.now()
        year = now.year
        return f"{year}-{year+1}"
