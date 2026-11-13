# src/display_log_handler.py
import time
import math
import logging
import os
from threading import Timer
from dotenv import load_dotenv
from collections import deque

load_dotenv()  # in case DISPLAY_SERVER_URL is in a .env file

try:
    from display_client import DisplayClient  # type: ignore
except Exception:
    DisplayClient = None  # we'll fallback to raw requests if needed

import requests  # fallback path

_UNMANAGED = object()

class DisplayPushHandler(logging.Handler):
    """
    Logging handler that streams recent log lines to the display server
    as a bottom-panel text overlay. It rate-limits pushes and uses a TTL
    so the overlay auto-hides after inactivity. Bursts get a trailing push;
    accepted responses keep the transcript visible through playback.
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
        self._last_push = float('-inf')
        self._muted_until = 0.0  # backoff window after an error
        self._timer = None
        self._responding = False
        self._expires_at = None
        self._stopped = False
        self._followup_managed = False
        self._hidden = False
        self._pending_clear = False

        # client choice
        if DisplayClient is not None:
            self._client = DisplayClient(self.base_url, timeout=self.timeout)
        else:
            self._client = None
            self._session = requests.Session()

        # default minimalist formatter if caller didn't set one
        if self.formatter is None:
            self.setFormatter(logging.Formatter("%(asctime)s :: %(message)s", datefmt="%H:%M:%S"))

    def follow_conversation(self):
        """Make the listening state, rather than log traffic, own visibility."""
        with self.lock:
            self._followup_managed = True
            self.hide_conversation()

    def hide_conversation(self):
        with self.lock:
            self._hidden = True
            self._responding = False
            self._expires_at = None
            self._lines.clear()
            self._pending_clear = True
            self._request_push(force=True)

    # Only DISPLAY and above? Set handler level outside or override .filter here if you want.
    # e.g., in hal.py: handler.setLevel(DISPLAY_LEVEL)

    def emit(self, record: logging.LogRecord) -> None:
        if self._stopped or getattr(record, 'skip_display', False):
            return
        try:
            line = self.format(record)
        except Exception:
            # never let formatting crash the app
            return

        with self.lock:
            if self._followup_managed and self._hidden:
                return  # Background diagnostics must not look like an open listening window.
            level = ('error' if record.levelno >= logging.ERROR else
                     'warning' if record.levelno >= logging.WARNING else 'conversation')
            self._lines.append({'text': line, 'level': level})
            if not self._followup_managed:
                self._expires_at = time.monotonic() + self.ttl_secs if self.ttl_secs else None
            self._request_push()

    def begin_response(self):
        """Hold existing conversation text while capturing/processing/speaking."""
        with self.lock:
            self._hidden = False
            self._pending_clear = False
            self._responding = True
            if self._followup_managed and self._lines:
                self._request_push(force=True)

    def begin_capture(self):
        self.begin_response()

    def end_response(self, *, followup_remaining=_UNMANAGED):
        """Start the inactivity timer after playback, including failure paths."""
        with self.lock:
            if followup_remaining is not _UNMANAGED:
                if followup_remaining is None or followup_remaining <= 0:
                    self.hide_conversation()
                    return
                self._responding = False
                self._expires_at = time.monotonic() + followup_remaining
                if self._lines:
                    self._request_push(force=True)
                return
            if not self._responding:
                return
            self._responding = False
            self._expires_at = time.monotonic() + self.ttl_secs if self.ttl_secs else None
            if self._lines:
                self._request_push(force=True)

    def _cancel_timer(self):
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _schedule(self, delay):
        self._cancel_timer()

        def send_pending():
            with self.lock:
                if self._timer is not timer or self._stopped:
                    return
                self._timer = None
                self._request_push()

        timer = Timer(delay, send_pending)
        timer.daemon = True
        self._timer = timer
        timer.start()

    def _request_push(self, *, force=False):
        if self._stopped:
            return
        if self._hidden and not self._pending_clear:
            return
        now = time.monotonic()
        if not self._responding and self._expires_at is not None and now >= self._expires_at:
            self._cancel_timer()  # A recovered server must not resurrect an old transcript.
            return
        delay = max(0, self._muted_until - now,
                    0 if force else self._last_push + self.min_push_interval - now)
        if delay:
            self._schedule(delay)
        else:
            self._cancel_timer()
            self._push(now)

    def close(self):
        with self.lock:
            self._stopped = True
            self._cancel_timer()
        super().close()

    # ---- internals ----
    def _compose_text(self) -> str:
        return ''.join(line['text'] for line in self._compose_lines())

    def _compose_lines(self) -> list[dict]:
        # Trim the same plain-text tail as before, retaining each record's level
        # even when the character boundary falls inside a multiline warning.
        size = sum(len(line['text']) for line in self._lines) + max(0, len(self._lines) - 1)
        discard = max(0, size - self.max_chars)
        result = [{'text': '…\n', 'level': 'conversation'}] if discard else []
        for index, line in enumerate(self._lines):
            text = ('\n' if index else '') + line['text']
            if discard >= len(text):
                discard -= len(text)
                continue
            result.append({'text': text[discard:], 'level': line['level']})
            discard = 0
        return result

    def _push(self, now: float) -> None:
        payload_lines = self._compose_lines()
        payload_text = ''.join(line['text'] for line in payload_lines)
        ttl = (None if self._responding or self._expires_at is None
               else max(1, math.ceil(self._expires_at - now)))
        try:
            if self._pending_clear:
                if self._client is not None:
                    self._client.clear(key=self.key)
                else:
                    r = self._session.post(f'{self.base_url}/api/clear',
                                           json={'key': self.key}, timeout=self.timeout)
                    r.raise_for_status()
                self._pending_clear = False
            elif self._client is not None:
                # Use the nice wrapper
                self._client.text(
                    payload_text,
                    text_lines=payload_lines,
                    on=self.slots,
                    priority=self.priority,
                    ttl=ttl,
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
                        "text_lines": payload_lines,
                        "slots": list(self.slots),
                        "priority": self.priority,
                        "ttl_secs": ttl,
                        "fullscreen": False,
                        "key": self.key,
                        "fit": "cover",
                        "bg": "#000",
                    },
                    timeout=self.timeout,
                )
                r.raise_for_status()

            self._last_push = time.monotonic()
            if self._muted_until:
                logging.getLogger('HAL').info('Display transcript delivery recovered.',
                                              extra={'skip_display': True})
            self._muted_until = 0.0
            logging.getLogger('HAL').debug('Display transcript sent: chars=%d; ttl=%s.',
                                           len(payload_text), ttl, extra={'skip_display': True})
            if self._followup_managed and self._responding and not self._hidden:
                # Keep the server's idle reaper from hiding a long-running turn.
                # The timer is replaced by a deadline or clear when the turn ends.
                self._schedule(30.)

        except Exception as exc:
            # brief backoff to avoid spamming errors when server is down
            if not self._muted_until:
                logging.getLogger('HAL').warning('Display transcript delivery failed (%s); retrying in 5 seconds.',
                                                 type(exc).__name__, extra={'skip_display': True})
            self._muted_until = time.monotonic() + 5.0
            self._schedule(5.0)
