# src/display_log_handler.py
import time
import logging
import os
from dotenv import load_dotenv
from collections import deque

load_dotenv()  # in case DISPLAY_SERVER_URL is in a .env file

try:
    from display_client import DisplayClient  # type: ignore
except Exception:
    DisplayClient = None  # we'll fallback to raw requests if needed

import requests  # fallback path

class DisplayPushHandler(logging.Handler):
    """
    Logging handler that streams recent log lines to the display server
    as a bottom-panel text overlay. It rate-limits pushes and uses a TTL
    so the overlay auto-hides after inactivity.
    """
    def __init__(
        self,
        base_url: str | None = None,
        *,
        slots=("bottom",),
        priority: int = 70,
        key: str = "logs",
        ttl_secs: int = 30,
        max_lines: int = 80,
        max_chars: int = 4000,
        min_push_interval: float = 0.25,
        timeout: float = 0.8,
    ):
        super().__init__()
        self.base_url = (base_url or os.getenv("DISPLAY_SERVER_URL") or "http://127.0.0.1:8000").rstrip("/")
        self.slots = tuple(slots)
        self.priority = priority
        self.key = key
        self.ttl_secs = ttl_secs
        self.max_lines = max_lines
        self.max_chars = max_chars
        self.min_push_interval = min_push_interval
        self.timeout = timeout

        self._lines = deque(maxlen=max_lines)
        self._last_push = 0.0
        self._muted_until = 0.0  # backoff window after an error

        # client choice
        if DisplayClient is not None:
            self._client = DisplayClient(self.base_url, timeout=self.timeout)
        else:
            self._client = None
            self._session = requests.Session()

        # default minimalist formatter if caller didn't set one
        if self.formatter is None:
            self.setFormatter(logging.Formatter("%(asctime)s :: %(message)s", datefmt="%H:%M:%S"))

    # Only DISPLAY and above? Set handler level outside or override .filter here if you want.
    # e.g., in hal.py: handler.setLevel(DISPLAY_LEVEL)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
        except Exception:
            # never let formatting crash the app
            return

        self._lines.append(line)
        now = time.time()

        # Rate-limit & error backoff
        if now < self._muted_until:
            return
        if (now - self._last_push) < self.min_push_interval:
            return

        self._push(now)

    # ---- internals ----
    def _compose_text(self) -> str:
        text = "\n".join(self._lines)
        if len(text) > self.max_chars:
            # keep tail, show an ellipsis marker
            text = "…\n" + text[-self.max_chars:]
        return text

    def _push(self, now: float) -> None:
        payload_text = self._compose_text()
        try:
            if self._client is not None:
                # Use the nice wrapper
                self._client.text(
                    payload_text,
                    on=self.slots,
                    priority=self.priority,
                    ttl=self.ttl_secs,
                    key=self.key,
                    bg="#000",
                )
            else:
                # Fallback raw HTTP
                r = self._session.post(
                    f"{self.base_url}/api/push",
                    json={
                        "type": "text",
                        "text": payload_text,
                        "slots": list(self.slots),
                        "priority": self.priority,
                        "ttl_secs": self.ttl_secs,
                        "fullscreen": False,
                        "key": self.key,
                        "fit": "cover",
                        "bg": "#000",
                    },
                    timeout=self.timeout,
                )
                r.raise_for_status()

            self._last_push = now

        except Exception:
            # brief backoff to avoid spamming errors when server is down
            self._muted_until = now + 5.0
