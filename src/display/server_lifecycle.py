"""
server_lifecycle.py
Start/stop the FastAPI display server (display.display_server:app) in a
background thread via uvicorn. Safe to import from anywhere.

- Auto-starts only if DISPLAY_SERVER_URL points at localhost.
- No-ops if the server is already running.
- Registers an atexit hook to stop the server.
"""

from __future__ import annotations
import atexit
import threading
import time
from typing import Optional, Tuple
from urllib.parse import urlparse

try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore

try:
    import uvicorn
except Exception:  # pragma: no cover
    uvicorn = None  # type: ignore


class DisplayServerManager:
    def __init__(
        self,
        url: str = "http://127.0.0.1:8000",
        app_import_path: str = "display.display_server:app",
        log_level: str = "info",
        logger=None,
        register_atexit: bool = True,
    ) -> None:
        self.url = url.rstrip("/")
        self.app_import_path = app_import_path
        self.log_level = log_level
        self.logger = logger
        self._server: Optional["uvicorn.Server"] = None
        self._thread: Optional[threading.Thread] = None
        if register_atexit:
            atexit.register(self.stop)

    # ---------------------------- helpers ---------------------------- #
    def _log(self, level: str, msg: str) -> None:
        if self.logger is not None:
            getattr(self.logger, level, self.logger.info)(msg)
        else:
            print(f"[display:{level}] {msg}")

    def _parse_bind(self) -> Tuple[str, int]:
        p = urlparse(self.url)
        host = p.hostname or "127.0.0.1"
        port = p.port or 8000
        return host, port

    @staticmethod
    def _is_loopback(host: str) -> bool:
        return host in {"127.0.0.1", "localhost", "0.0.0.0"}

    def _reachable(self, timeout: float = 0.5) -> bool:
        if requests is None:
            return False
        try:
            r = requests.get(f"{self.url}/api/state", timeout=timeout)
            return r.ok
        except Exception:
            return False

    # ----------------------- public API ----------------------------- #
    def start(self, wait_seconds: float = 2.0) -> None:
        """Start uvicorn in a daemon thread if (a) local URL and (b) not already running."""
        if uvicorn is None:
            self._log("warning", "uvicorn not installed; skipping display server auto-start.")
            return

        host, port = self._parse_bind()

        # If URL points remote, assume someone else runs the server.
        if not self._is_loopback(host):
            self._log("info", f"DISPLAY_SERVER_URL is remote ({host}); not auto-starting.")
            return

        # Already up?
        if self._reachable(timeout=0.3):
            self._log("info", f"Display server already running at {self.url}")
            return

        # Lazy import app
        try:
            module_path, attr = self.app_import_path.split(":")
            mod = __import__(module_path, fromlist=[attr])
            app = getattr(mod, attr)
        except Exception as e:  # pragma: no cover
            self._log("warning", f"Could not import {self.app_import_path}: {e}")
            return

        config = uvicorn.Config(app, host=host, port=port, log_level=self.log_level)
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, name="DisplayServer", daemon=True)
        thread.start()
        self._server = server
        self._thread = thread

        # Wait briefly for readiness (best effort)
        deadline = time.time() + wait_seconds
        while time.time() < deadline:
            if self._reachable(timeout=0.2):
                self._log("info", f"Display server started at {self.url}")
                return
            time.sleep(0.1)
        self._log("warning", "Display server did not respond in time; continuing anyway.")

    def stop(self) -> None:
        """Signal uvicorn to stop and join the thread (best effort)."""
        try:
            if self._server is not None:
                self._server.should_exit = True
            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=3)
        except Exception:
            pass
        finally:
            self._server = None
            self._thread = None
