# calendar_api.py
from abc import ABC, abstractmethod
from typing import List, Optional
import datetime

from icloud_service import ICloudService


class CalendarBackend(ABC):
    """Abstract base class for calendar backends (iCloud, Google, etc.)."""

    def dispatch(self, command: str, params: Optional[list] = None):
        """
        Map a command and its parameters to the appropriate backend method.
        HAL will provide the parsed command and parameters.
        """
        params = params or []

        try:
            if command == "calendar_search":
                # expects one parameter: query
                return self.search_events(params[0])

            elif command == "calendar_next_event":
                # optional parameter: calendar_name
                cal_name = self._resolve_calendar_name(params[0] if params else None)
                return self.next_event(cal_name)

            elif command == "calendar_on_date":
                # expects at least date_expr, optional calendar_name
                date_expr = params[0]
                cal_name = self._resolve_calendar_name(params[1] if len(params) > 1 else None)

                # Special routing for "this week" and "next week"
                normalized_expr = date_expr.lower().strip()
                if normalized_expr == "this week":
                    return self.events_this_week(cal_name)
                elif normalized_expr in ["next week","this coming week"]:
                    return self.events_next_week(cal_name)
                else:
                    return self.events_on_date(date_expr, cal_name)

            else:
                raise ValueError(f"Unknown calendar command: {command}")
        except ValueError as e:
            return {'error':f"I'm sorry, there was an error when accessing your calendar: {e}"}
        

    @abstractmethod
    def events_this_week(self, calendar: Optional[str] = None) -> List[dict]:
        pass

    @abstractmethod
    def events_next_week(self, calendar: Optional[str] = None) -> List[dict]:
        pass

    @abstractmethod
    def search_events(self, query: str) -> List[dict]:
        pass

    @abstractmethod
    def next_event(self, calendar: Optional[str] = None) -> Optional[dict]:
        pass

    @abstractmethod
    def events_on_date(self, date_expr: str, calendar: Optional[str] = None) -> List[dict]:
        pass

    @abstractmethod
    def _resolve_calendar_name(self, requested_name: str) -> str:
        """Return properly-cased calendar name or raise ValueError if not found."""
        pass


class ICloudCalendar(CalendarBackend):
    """iCloud Calendar adapter that wraps ICloudService."""

    def __init__(self, icloud_service: Optional[ICloudService] = None):
        # Either use a provided service or create one on the fly
        self.service = icloud_service or ICloudService()

    def events_this_week(self, calendar: Optional[str] = None) -> List[dict]:
        return self.service.events_this_week(calendar)

    def events_next_week(self, calendar: Optional[str] = None) -> List[dict]:
        return self.service.events_next_week(calendar)

    def search_events(self, query: str) -> List[dict]:
        return self.service.search_events(query)

    def next_event(self, calendar: Optional[str] = None) -> Optional[dict]:
        return self.service.next_event(calendar)

    def events_on_date(self, date_expr: str, calendar: Optional[str] = None) -> List[dict]:
        return self.service.events_on_date(date_expr, calendar)
    
    def _resolve_calendar_name(self, requested_name: str) -> str:
        """Return the properly-cased calendar name, or raise ValueError if not found."""
        if not requested_name:
            return None
        calendars = self.service.api.calendar.get_calendars()
        for cal in calendars:
            if cal.get("title", "").lower() == requested_name.lower():
                return cal["title"]
        raise ValueError(f"Calendar named '{requested_name}' was not found.")



# mini test suite if run as __main__
if __name__ == "__main__":
    calendar = ICloudCalendar()  # spins up its own ICloudService

    # Mini test dispatcher calls
    test_cases = [
        # Search for a keyword
        ("calendar_search", ["Production Meeting"]),

        # Next event in default calendar
        ("calendar_next_event", []),

        # Next event in specific calendar
        ("calendar_next_event", ["Composing"]),

        # Events on a specific date
        ("calendar_on_date", ["next Wednesday"]),

        # Events on a specific date in a specific calendar
        ("calendar_on_date", ["next Monday", "Composing"]),

        # Events this week (intercepted by dispatcher)
        ("calendar_on_date", ["this week"]),

        # Events next week (intercepted by dispatcher)
        ("calendar_on_date", ["next week", "composing"]),

        # Events next week on calendar that does not exist
        ("calendar_on_date", ["next week", "gfngkfldsngfdskfj"]),
    ]

    for cmd, params in test_cases:
        print(f"\n--- Dispatching Command: {cmd} ---")
        print(f"Parameters: {params}")
        results = calendar.dispatch(cmd, params)
        if not results:
            print("No events found.")
        else:
            # Wrap single dict results in a list for consistency
            results_list = results if isinstance(results, list) else [results]
            for e in results_list:
                if 'error' in e.keys():
                    print(e)
                else:
                    print(f"- {e['title']} on {e['start_str']}")

