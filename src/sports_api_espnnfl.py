import requests
import os
from datetime import datetime, timezone
from sports_api_base import BaseSportsAPI
import pytz  # pip install pytz

class ESPNnflAPI(BaseSportsAPI):
    """NFL data via ESPN unofficial JSON endpoints."""

    BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
    STANDINGS_URL = "https://cdn.espn.com/core/nfl/standings"

    def __init__(self):
        tz_name = os.getenv("TIMEZONE", "UTC")
        self.local_tz = pytz.timezone(tz_name)

    def _current_season(self):
        """Return the NFL season year. If Jan–Mar, it's the previous year's season."""
        now = datetime.now(self.local_tz)
        if now.month < 4:
            return now.year - 1
        return now.year

    def _fetch_week(self, season_year, season_type, week):
        """Fetch events for a given week/season type. Returns [] if unavailable."""
        params = {"seasontype": season_type, "week": week}
        if season_year:
            params["season"] = season_year
        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=10)
            resp.raise_for_status()
            return resp.json().get("events", [])
        except requests.RequestException:
            return []

    def _fetch_full_season(self, season_year=None):
        """Fetch all games for the season (regular + postseason if available)."""
        all_events = []

        # Regular season: weeks 1–18
        for wk in range(1, 19):
            events = self._fetch_week(season_year, season_type=2, week=wk)
            all_events.extend(events)

        # Postseason: try weeks 1–5 (Wildcard → Super Bowl)
        for wk in range(1, 6):
            events = self._fetch_week(season_year, season_type=3, week=wk)
            if not events:
                break
            all_events.extend(events)

        return all_events

    def schedule(self, team_or_league):
        """Return full season schedule for a team or league ('NFL')."""
        events = self._fetch_full_season()
        results = []

        for event in events:
            game = self._parse_event(event)

            # Filter by league or team
            if team_or_league.upper() == "NFL":
                results.append(game)
            else:
                if team_or_league.upper() in [game["home_team"].upper(), game["away_team"].upper()]:
                    results.append(game)

        # Sort by date
        results.sort(key=lambda g: datetime.fromisoformat(g["date"]))
        return results

    def next_game(self, team_or_league):
        """Return the next upcoming game for a team, or league-wide if 'NFL'."""
        events = self.schedule(team_or_league)
        now_utc = datetime.now(timezone.utc)

        for game in events:
            # Convert local-aware datetime back to UTC for comparison
            game_time_local = datetime.fromisoformat(game["date"])
            game_time_utc = game_time_local.astimezone(timezone.utc)

            if game_time_utc >= now_utc:
                return game

        return None
    
    def find_game(self, team1, team2):
        """Return all games between two teams in chronological order for the current season."""

        team1_upper = team1.upper()
        team2_upper = team2.upper()
        
        events = self.schedule(team1)  # get team1's schedule
        matched_games = []

        for game in events:
            home = game["home_team"].upper()
            away = game["away_team"].upper()
            if (team1_upper in [home, away] and team2_upper in [home, away]):
                matched_games.append(game)

        # Already stored in local timezone by _parse_event(), just sort chronologically
        matched_games.sort(key=lambda g: g["date"])
        return matched_games

    def standings(self, league=None):
        """Return current NFL standings by conference/division."""
        try:
            resp = requests.get(self.STANDINGS_URL, params={"xhr": 1}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            return {"error": f"Failed to fetch standings: {e}"}

        standings_data = []

        # Navigate into conferences
        league_groups = data.get("content", {}).get("standings", {}).get("groups", [])

        for conference in league_groups:  # e.g., AFC / NFC
            conf_name = conference.get("name")
            divisions = []
            for division in conference.get("groups", []):  # e.g., AFC East, AFC North...
                div_name = division.get("name")
                teams = []
                for entry in division.get("standings", {}).get("entries", []):
                    team_info = entry.get("team", {})
                    stats = {s["name"]: s.get("displayValue") for s in entry.get("stats", [])}

                    teams.append({
                        "team": team_info.get("displayName"),
                        "abbrev": team_info.get("abbreviation"),
                        "wins": stats.get("wins"),
                        "losses": stats.get("losses"),
                        "ties": stats.get("ties"),
                        "pct": stats.get("winPercent"),
                        "streak": stats.get("streak"),
                        "pointsFor": stats.get("pointsFor"),
                        "pointsAgainst": stats.get("pointsAgainst"),
                        "diff": stats.get("differential"),
                        "logo": team_info.get("logos", [{}])[0].get("href"),
                    })

                divisions.append({
                    "division": div_name,
                    "teams": teams
                })

            standings_data.append({
                "conference": conf_name,
                "divisions": divisions
            })

        return standings_data

    def _parse_event(self, event):
        """Extract relevant info from ESPN event JSON."""
        competition = event["competitions"][0]
        competitors = competition["competitors"]

        home = next(c for c in competitors if c["homeAway"] == "home")
        away = next(c for c in competitors if c["homeAway"] == "away")

        # Broadcasts
        broadcasts = []
        for b in competition.get("broadcasts", []):
            if "names" in b:
                broadcasts.extend(b["names"])
            elif "media" in b and "shortName" in b["media"]:
                broadcasts.append(b["media"]["shortName"])

        # Convert event date to local timezone for HAL
        event_dt_utc = datetime.fromisoformat(event["date"].replace("Z", "+00:00"))
        event_dt_local = event_dt_utc.astimezone(self.local_tz)

        return {
            "date": event_dt_local.isoformat(),  # local-aware ISO string
            "home_team": home["team"]["displayName"],
            "away_team": away["team"]["displayName"],
            "home_score": home.get("score"),
            "away_score": away.get("score"),
            "status": competition["status"]["type"]["name"],
            "venue": competition.get("venue", {}).get("fullName"),
            "week": competition.get("week", {}).get("number"),
            "season_type": competition.get("season", {}).get("type"),
            "broadcasts": broadcasts,
            "links": {
                "boxscore": next((l["href"] for l in competition.get("links", []) if l["text"] == "Boxscore"), None),
                "recap": next((l["href"] for l in competition.get("links", []) if l["text"] == "Recap"), None),
            },
        }
