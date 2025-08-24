import re
from sports_api_sportsdb import TheSportsDBAPI
from sports_api_amfootball import APISportsAmericanFootballAPI
from sports_api_espnnfl import ESPNnflAPI

class SportsRouter:
    def __init__(self):
        self.backends = {
            # "NFL": APISportsAmericanFootballAPI(),
            "NFL": ESPNnflAPI(),
            "DEFAULT": TheSportsDBAPI(),
        }

        # Keywords → synonyms for fuzzy match
        self.nfl_aliases = {
            "ARIZONA CARDINALS": ["CARDINALS", "CARDS"],
            "ATLANTA FALCONS": ["FALCONS", "DIRTY BIRDS"],
            "BALTIMORE RAVENS": ["RAVENS"],
            "BUFFALO BILLS": ["BILLS"],
            "CAROLINA PANTHERS": ["PANTHERS"],
            "CHICAGO BEARS": ["BEARS"],
            "CINCINNATI BENGALS": ["BENGALS"],
            "CLEVELAND BROWNS": ["BROWNS"],
            "DALLAS COWBOYS": ["COWBOYS", "BOYS"],
            "DENVER BRONCOS": ["BRONCOS"],
            "DETROIT LIONS": ["LIONS"],
            "GREEN BAY PACKERS": ["PACKERS", "PACK"],
            "HOUSTON TEXANS": ["TEXANS"],
            "INDIANAPOLIS COLTS": ["COLTS"],
            "JACKSONVILLE JAGUARS": ["JAGUARS", "JAGS"],
            "KANSAS CITY CHIEFS": ["CHIEFS"],
            "LAS VEGAS RAIDERS": ["RAIDERS"],
            "LOS ANGELES CHARGERS": ["CHARGERS", "BOLTS"],
            "LOS ANGELES RAMS": ["RAMS"],
            "MIAMI DOLPHINS": ["DOLPHINS", "FINS"],
            "MINNESOTA VIKINGS": ["VIKINGS", "VIKES"],
            "NEW ENGLAND PATRIOTS": ["PATRIOTS", "PATS"],
            "NEW ORLEANS SAINTS": ["SAINTS"],
            "NEW YORK GIANTS": ["GIANTS", "G-MEN", "GMEN"],
            "NEW YORK JETS": ["JETS"],
            "PHILADELPHIA EAGLES": ["EAGLES"],
            "PITTSBURGH STEELERS": ["STEELERS"],
            "SAN FRANCISCO 49ERS": ["49ERS", "NINERS", "'9ERS"],
            "SEATTLE SEAHAWKS": ["SEAHAWKS", "HAWKS"],
            "TAMPA BAY BUCCANEERS": ["BUCCANEERS", "BUCS", "BUCCS"],
            "TENNESSEE TITANS": ["TITANS"],
            "WASHINGTON COMMANDERS": ["COMMANDERS", "COMMIES"],
        }

        # Global league terms
        self.league_aliases = {
            "NFL": ["NFL", "NATIONAL FOOTBALL LEAGUE", "AMERICAN FOOTBALL", "FOOTBALL"],
        }

    def _normalize(self, text: str) -> str:
        return re.sub(r"[^A-Z0-9 ]", "", text.upper())

    def _resolve_nfl_team_or_league(self, text: str):
        norm = self._normalize(text)

        # Check league first
        for league, aliases in self.league_aliases.items():
            if norm in aliases or norm == league:
                return league

        # Check teams
        for team, aliases in self.nfl_aliases.items():
            if norm == team:
                return team
            if any(norm == self._normalize(alias) for alias in aliases):
                return team
            # Partial match (e.g. "VIKES" inside "GO VIKES")
            if any(alias in norm for alias in [self._normalize(a) for a in aliases]):
                return team

        return None

    def _choose_backend(self, team_or_league: str):
        match = self._resolve_nfl_team_or_league(team_or_league)
        if match:
            return self.backends["NFL"], match
        return self.backends["DEFAULT"], team_or_league

    def dispatch(self, action: str, target: str, target2: str = None):
        """
        Dispatch API calls.
        - For 'find_game', target2 must be provided.
        - For other actions, only target is used.
        """
        backend, resolved_target = self._choose_backend(target)

        if action == "find_game":
            if not target2:
                raise ValueError("find_game requires two teams")
            _, resolved_target2 = self._choose_backend(target2)
            return backend.dispatch(action, [resolved_target, resolved_target2])

        return backend.dispatch(action, [resolved_target])


if __name__ == "__main__":
    import json

    print("=== SportsRouter Smoke Tests ===")

    router = SportsRouter()

    try:
        # Test 1: Team alias resolution (Vikings)
        print("\n[TEST 1] Next game for 'Vikings'")
        resp = router.dispatch("next_game", "Vikings")
        print(json.dumps(resp, indent=2)[:500], "...")

        # Test 2: Full team name
        print("\n[TEST 2] Schedule for 'Minnesota Vikings'")
        resp = router.dispatch("schedule", "Minnesota Vikings")
        print(f"Total games returned: {len(resp)}")
        print(json.dumps(resp, indent=2)[:500], "...")

        # Test 3: League schedule
        print("\n[TEST 3] Schedule for 'NFL'")
        resp = router.dispatch("schedule", "NFL")
        print(f"Total games returned: {len(resp)}")
        print(json.dumps(resp[:2], indent=2))  # preview

        # Test 4: Standings
        print("\n[TEST 4] Standings for 'NFL'")
        resp = router.dispatch("standings", "NFL")
        print(json.dumps(resp, indent=2)[:500], "...")

        # Test 5: Find specific game
        print("\n[TEST 5] Vikings vs Packers")
        resp = router.dispatch("find_game", "Vikings", "Packers")
        print(json.dumps(resp, indent=2)[:1000], "...")

    except Exception as e:
        print("\n[ERROR]", e)
