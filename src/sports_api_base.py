class BaseSportsAPI:
    """Abstract base class for sports API integrations."""

    def dispatch(self, command: str, params: list):
        try:
            if command == "next_game":
                if not params:
                    raise ValueError("Missing team or league parameter.")
                return self.next_game(params[0])

            elif command == "schedule":
                if not params:
                    raise ValueError("Missing team or league parameter.")
                return self.schedule(params[0])
            
            elif command == "find_game":
                if not params:
                    raise ValueError("Missing team1 and team2 parameters.")
                if len(params) != 2:
                    raise ValueError("parameters must contain two teams")
                return self.find_game(params[0],params[1])

            elif command == "standings":
                if not params:
                    raise ValueError("Missing team or league parameter.")
                return self.standings(params[0])

            else:
                raise ValueError(f"Unknown sports command: {command}")

        except Exception as e:
            return {"error": f"I'm sorry, there was an error accessing sports data: {e}"}

    # Abstract methods
    def next_game(self, name: str):
        raise NotImplementedError

    def schedule(self, name: str):
        raise NotImplementedError
    
    def find_game(self, team1: str, team2: str):
        raise NotImplementedError

    def standings(self, league: str):
        raise NotImplementedError
