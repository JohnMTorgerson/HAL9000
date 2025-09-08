
import os
from dotenv import load_dotenv
import requests
from typing import Iterable, Optional, Literal, Dict, Any

load_dotenv()  # in case DISPLAY_SERVER_URL is in a .env file

PanelSlot = Literal["top", "bottom"]
ContentType = Literal["image", "text", "url"]

DISPLAY_SERVER = os.getenv("DISPLAY_SERVER_URL", "http://127.0.0.1:8000")

class DisplayClient:
    def __init__(self, base_url: str = DISPLAY_SERVER, timeout: float = 3.0):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self._s = requests.Session()

    # -------- core helpers --------
    def push(
        self,
        *,
        type: ContentType,
        text: Optional[str] = None,
        src: Optional[str] = None,
        slots: Optional[Iterable[PanelSlot]] = None,
        priority: int = 50,
        ttl_secs: Optional[int] = 120,
        fullscreen: bool = False,
        key: Optional[str] = None,
        fit: Literal["cover","contain"] = "cover",
        bg: str = "#000",
    ) -> Dict[str, Any]:
        payload = {
            "type": type,
            "text": text,
            "src": src,
            "slots": list(slots) if slots is not None else None,
            "priority": priority,
            "ttl_secs": ttl_secs,
            "fullscreen": fullscreen,
            "key": key,
            "fit": fit,
            "bg": bg,
        }
        r = self._s.post(f"{self.base}/api/push", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def clear(
        self,
        *,
        all: bool = False,
        key: Optional[str] = None,
        ids: Optional[Iterable[str]] = None,
        slot: Optional[Literal["top","bottom","both"]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "all": all,
            "key": key,
            "ids": list(ids) if ids is not None else None,
            "slot": slot,
        }
        r = self._s.post(f"{self.base}/api/clear", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    # -------- conveniences --------
    def text(
        self, msg: str, *, on: Iterable[PanelSlot]=("top",), priority: int=50,
        ttl: Optional[int]=120, key: Optional[str]=None, fullscreen: bool=False,
        bg: str="#000"
    ):
        return self.push(type="text", text=msg, slots=on, priority=priority,
                         ttl_secs=ttl, key=key, fullscreen=fullscreen, bg=bg)

    def image(
        self, src: str, *, on: Iterable[PanelSlot]=("top",), priority: int=50,
        ttl: Optional[int]=120, key: Optional[str]=None, fit: Literal["cover","contain"]="cover",
        fullscreen: bool=False, bg: str="#000"
    ):
        return self.push(type="image", src=src, slots=on, priority=priority,
                         ttl_secs=ttl, key=key, fullscreen=fullscreen, fit=fit, bg=bg)

    def url(
        self, src: str, *, on: Iterable[PanelSlot]=("top",), priority: int=80,
        ttl: Optional[int]=60, key: Optional[str]=None, fullscreen: bool=False, bg: str="#000"
    ):
        return self.push(type="url", src=src, slots=on, priority=priority,
                         ttl_secs=ttl, key=key, fullscreen=fullscreen, bg=bg)
