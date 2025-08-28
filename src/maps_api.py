# maps_api.py

from typing import Any, Dict, List, Union
import maps_api_googleplaces

SearchResult = Dict[str, Union[str, float, bool, List[str]]]


class MapsRouter:
    """Abstraction layer for maps-related commands (search, directions, etc.)."""

    def __init__(self, backend: str = "google"):
        # In the future we could support other backends (e.g. Yelp).
        self.backend = backend

    def dispatch(self, command: str, params: Dict[str, Any]) -> Union[List[SearchResult], Dict[str, str]]:
        """
        Dispatch a maps-related command.

        Args:
            command: the action to perform, e.g. "search".
            params: dictionary of parameters for the command.

        Returns:
            - On success: a list of SearchResult dicts.
            - On error: a dict with {"error": "<human readable message>"}.
        """
        if command == "search":
            query = params.get("query")
            radius = params.get("radius", 5000)

            if not query:
                return {"error": "Missing required parameter: 'query'."}

            if self.backend == "google":
                return maps_api_googleplaces.search(query, radius=radius)

            # Placeholder for other backends
            return {"error": f"Unsupported backend '{self.backend}' for command '{command}'."}

        return {"error": f"Unknown maps command: '{command}'"}

if __name__ == "__main__":
    import json
    router = MapsRouter()

    print("=== Maps API Test ===")

    try:
        # Test 1: Search for a hardware store
        print("\n[TEST 1] Nearby hardware stores")
        results = router.dispatch("search", {"query": "hardware store", "radius": 3000})
        print(json.dumps(results, indent=2))

        # Test 2: Search for a coffee shop
        print("\n[TEST 2] Nearby coffee shops")
        results = router.dispatch("search", {"query": "coffee"})
        print(json.dumps(results, indent=2))

        # Test 3: Missing query
        print("\n[TEST 3] Missing query")
        results = router.dispatch("search", {})
        print(json.dumps(results, indent=2))

        # Test 3: Empty query
        print("\n[TEST 4] Empty query")
        results = router.dispatch("search", {"query":""})
        print(json.dumps(results, indent=2))

    except Exception as e:
        print("\n[ERROR]", str(e))