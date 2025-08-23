# icloud_service.py

import os
import sys
import datetime
import dateparser
from dotenv import load_dotenv
from pyicloud import PyiCloudService
import click


class ICloudService:
    def __init__(self, apple_id=None, app_password=None):
        load_dotenv()
        apple_id = apple_id or os.getenv("APPLE_ID")
        app_password = app_password or os.getenv("ICLOUD_PWD")

        if not apple_id or not app_password:
            raise ValueError("Missing APPLE_ID or ICLOUD_PWD in environment")

        self.api = PyiCloudService(apple_id, app_password)
        self.handle_authentication()

    def handle_authentication(self):
        """Handle iCloud 2FA/2SA if required."""
        api = self.api

        if api.requires_2fa:
            security_key_names = api.security_key_names
            if security_key_names:
                print(
                    f"Security key confirmation required. "
                    f"Plug in one of: {', '.join(security_key_names)}"
                )
                devices = api.fido2_devices
                for idx, dev in enumerate(devices, start=1):
                    print(f"{idx}: {dev}")
                choice = click.prompt(
                    "Select a FIDO2 device by number",
                    type=click.IntRange(1, len(devices)),
                    default=1,
                )
                selected_device = devices[choice - 1]
                api.confirm_security_key(selected_device)
            else:
                print("Two-factor authentication required.")
                code = input("Enter the code sent to your device: ")
                result = api.validate_2fa_code(code)
                if not result:
                    print("Failed to verify 2FA code")
                    sys.exit(1)

            if not api.is_trusted_session:
                print("Requesting trust...")
                api.trust_session()

        elif api.requires_2sa:
            print("Two-step authentication required. Your trusted devices:")
            devices = api.trusted_devices
            for i, device in enumerate(devices):
                label = device.get("deviceName", f"SMS to {device.get('phoneNumber')}")
                print(f"{i}: {label}")

            device_index = click.prompt("Which device?", default=0)
            device = devices[device_index]
            if not api.send_verification_code(device):
                print("Failed to send verification code")
                sys.exit(1)

            code = click.prompt("Enter validation code")
            if not api.validate_verification_code(device, code):
                print("Failed to verify code")
                sys.exit(1)

    # --------------------------
    # Calendar Functions
    # --------------------------

    def _find_calendar_by_name(self, name: str):
        """Return the calendar dict (from get_calendars) whose title matches name, or None."""
        calendars = self.api.calendar.get_calendars()
        name_norm = name.strip().lower()
        for cal in calendars:
            if cal.get("title", "").strip().lower() == name_norm:
                return cal
        return None

    @staticmethod
    def _event_calendar_guid(ev: dict):
        """Best-effort way to read the calendar GUID from an event dict across pyicloud versions."""
        for key in ("pGuid", "calendarGuid", "calendarId", "calendar_id"):
            if key in ev:
                return ev[key]
        return None
    
    @staticmethod
    def _parse_event_date(date_val):
        """Convert a pyicloud event date (str, datetime, or list) to a datetime object."""
        if isinstance(date_val, datetime.datetime):
            return date_val
        if isinstance(date_val, datetime.date):
            return datetime.datetime.combine(date_val, datetime.time.min)
        if isinstance(date_val, str):
            return dateparser.parse(date_val)
        if isinstance(date_val, list) and len(date_val) >= 6:
            # [YYYYMMDD, year, month, day, hour, minute, ...]
            try:
                return datetime.datetime(date_val[1], date_val[2], date_val[3], date_val[4], date_val[5])
            except Exception:
                return None
        return None
    
    def _normalize_event(self, ev: dict) -> dict:
        """Normalize an event dict with calendar info and human-readable dates."""
        cal_guid = self._event_calendar_guid(ev)
        cal_name = None
        if cal_guid:
            calendars = {c["guid"]: c["title"] for c in self.api.calendar.get_calendars()}
            cal_name = calendars.get(cal_guid)

        start_dt = ev.get("startDate")
        end_dt   = ev.get("endDate")

        # Ensure datetime objects
        start_dt = self._parse_event_date(start_dt)
        end_dt = self._parse_event_date(end_dt)

        start_str = start_dt.strftime("%A %B %d %I:%M %p") if start_dt else None
        end_str   = end_dt.strftime("%A %B %d %I:%M %p") if end_dt else None

        return {
            "title": ev.get("title"),
            "location": ev.get("location"),
            "description": ev.get("description"),
            "start": start_dt,
            "end": end_dt,
            "start_str": start_str,
            "end_str": end_str,
            "guid": ev.get("guid"),
            "calendar_guid": cal_guid,
            "calendar_name": cal_name,
            "raw": ev,
        }

    
    def _parse_date_expr(self, expr):
        """Accept datetime/date objects or natural language strings."""
        if isinstance(expr, (datetime.date, datetime.datetime)):
            return expr
        if isinstance(expr, str):

            expr = expr.strip().lower()
            today = datetime.date.today()

            # Handle "next <weekday>" explicitly
            weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            if expr.startswith("next "):
                wd_name = expr.split("next ", 1)[1].strip()
                if wd_name in weekdays:
                    target_wd = weekdays.index(wd_name)
                    days_ahead = (target_wd - today.weekday() + 7) % 7
                    days_ahead = 7 if days_ahead == 0 else days_ahead  # always at least next week
                    return today + datetime.timedelta(days=days_ahead)

            # Fallback to dateparser
            parsed = dateparser.parse(
                expr,
                settings={"PREFER_DATES_FROM": "future", "RELATIVE_BASE": datetime.datetime.now()},
            )
            if not parsed:
                raise ValueError(f"Couldn't parse date expression: {expr}")
            return parsed
        raise TypeError(f"Unsupported date expression type: {type(expr)}")

    def events_in_range(self, start_expr, end_expr, calendar=None):
        """Get events in a datetime range (supports natural language)."""
        start = self._parse_date_expr(start_expr)
        end   = self._parse_date_expr(end_expr)

        # iCloud wants datetimes (not just dates)
        if isinstance(start, datetime.date) and not isinstance(start, datetime.datetime):
            start = datetime.datetime.combine(start, datetime.time.min)
        if isinstance(end, datetime.date) and not isinstance(end, datetime.datetime):
            end = datetime.datetime.combine(end, datetime.time.max)

        all_events = self.api.calendar.get_events(start, end)
        if calendar:
            # Filter down to only the requested calendar
            calendars = self.api.calendar.get_calendars()
            cal_map = {c["title"]: c for c in calendars}
            if calendar not in cal_map:
                raise ValueError(f"Calendar '{calendar}' not found. Available: {list(cal_map)}")
            cal_guid = cal_map[calendar]["guid"]
            return [self._normalize_event(e) for e in all_events if self._event_calendar_guid(e) == cal_guid]

        return [self._normalize_event(e) for e in all_events]
    
    def events_on_date(self, date_expr, calendar=None):
        """Get all events on a specific date (natural language supported)."""
        parsed_date = self._parse_date_expr(date_expr)

        # Normalize to date if datetime
        if isinstance(parsed_date, datetime.datetime):
            parsed_date = parsed_date.date()

        start = datetime.datetime.combine(parsed_date, datetime.time.min)
        end   = datetime.datetime.combine(parsed_date, datetime.time.max)

        return self.events_in_range(start, end, calendar)
    
    def events_this_week(self, calendar=None):
        """Return events from today through the end of the current week (Sunday).
        If today is Sunday, include events through the end of next Sunday."""
        today = datetime.date.today()
        weekday = today.weekday()  # Monday = 0, Sunday = 6

        if weekday == 6:  # today is Sunday
            # include events through next Sunday
            end_day = today + datetime.timedelta(days=7)
        else:
            # end of this week
            end_day = today + datetime.timedelta(days=(6 - weekday))

        start_dt = datetime.datetime.combine(today, datetime.time.min)
        end_dt = datetime.datetime.combine(end_day, datetime.time.max)

        return self.events_in_range(start_dt, end_dt, calendar)

    def events_next_week(self, calendar=None):
        """Events in the next calendar week (Monday–Sunday after this week)."""
        today = datetime.date.today()
        weekday = today.weekday()  # Monday = 0, Sunday = 6

        # Days until next Monday
        days_until_monday = (7 - weekday) % 7
        if days_until_monday == 0:
            days_until_monday = 7  # if today is Monday, jump to next Monday

        start = today + datetime.timedelta(days=days_until_monday)
        end = start + datetime.timedelta(days=6)

        start_dt = datetime.datetime.combine(start, datetime.time.min)
        end_dt = datetime.datetime.combine(end, datetime.time.max)

        return self.events_in_range(start_dt, end_dt, calendar)

    def search_events(self, keyword, days_ahead=30, calendar=None):
        # print("search_events()")
        """Search for an event by keyword in any text field in the next N days."""
        start = datetime.datetime.now()
        end = start + datetime.timedelta(days=days_ahead)
        events = self.events_in_range(start, end, calendar)
        # print(f"events: {events}")

        keyword_lower = keyword.lower()

        def event_matches(ev):
            for key, value in ev.items():
                if isinstance(value, str) and keyword_lower in value.lower():
                    return True
            return False

        return [e for e in events if event_matches(e)]

    def next_event(self, calendar=None):
        """Get the very next event (optionally restricted to a calendar)."""
        start = datetime.datetime.now()
        end = start + datetime.timedelta(days=30)
        events = sorted(
            self.events_in_range(start, end, calendar),
            key=lambda e: e.get("start")
        )
        return events[0] if events else None

# --------------------------
# Tests
# --------------------------

if __name__ == "__main__":
    service = ICloudService()

    print("\n--- Next Week's Events ---")
    events = service.events_next_week()
    for e in events:
        print(e)

    print("\n--- Search for 'Production Meeting' ---")
    matches = service.search_events("Production Meeting")
    for e in matches:
        print(e)

    print("\n--- Next event on 'Composing' calendar ---")
    next_comp = service.next_event(calendar="Composing")
    if next_comp:
        print(e)
    else:
        print("No upcoming events on Composing calendar")

    query = "next Monday"
    print(f"\n--- Events on '{query}' ---")
    day_events = service.events_on_date(query)
    if day_events:
        for e in day_events:
            print(e)
    else:
        print("No events that day")