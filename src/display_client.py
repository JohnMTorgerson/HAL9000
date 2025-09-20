import os
from dotenv import load_dotenv
import requests
from typing import Iterable, Optional, Literal, Dict, Any
import json
from urllib.parse import quote

load_dotenv()

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

    # -------- API displays --------
    def map(
        self,
        places: list[dict],
        *,
        on=("top",),
        priority: int = 80,
        ttl: int | None = 120,
        key: str | None = "map",
        fullscreen: bool = False,
        center: tuple[float, float] | None = None,  # (lat, lon)
        zoom: int = 14,
        scale: float = 1.8,           # global style scale (labels, icons, line widths)
        uiscale: float | None = None, # UI controls scale (defaults to scale)
        pinscale: float | None = None,# pin/marker scale (defaults to uiscale or scale)
        maptiler_key: str | None = None,
        my_location: tuple[float, float] | None = None,  # (lat, lon); if None we read env
    ):
        """
        places: list of dicts like:
        {
            "name": "Bryant Hardware",
            "lat": 44.937805,
            "lon": -93.290298,
            "open_now": True,
            "distance_miles": 0.5,
            # optional: "label", "distance", "color", "size", "stroke", "strokeWidth", "id", "address"
        }
        """
        maptiler_key = maptiler_key or os.getenv("MAPTILER_API_KEY", "")
        s  = float(scale)
        ui = float(uiscale) if uiscale is not None else s
        ps = float(pinscale) if pinscale is not None else ui

        pins = []
        for p in places or []:
            lat = p.get("lat")
            lng = p.get("lng", p.get("lon") or p.get("longitude"))
            if lat is None or lng is None:
                continue

            # Normalize distance: prefer existing "distance" string, else format miles if present
            distance_str = None
            if "distance" in p and isinstance(p["distance"], str):
                distance_str = p["distance"]
            elif "distance_miles" in p:
                try:
                    distance_str = f"{float(p['distance_miles']):.1f} mi"
                except Exception:
                    pass

            # Build pin payload
            pin = {
                "lat": float(lat),
                "lng": float(lng),

                # label shown next to pin on the map (map.js uses this)
                "label": str(p.get("label") or p.get("name") or p.get("title") or ""),

                # extra metadata for future use
                "name": p.get("name") or p.get("title"),
                "open_now": bool(p.get("open_now")) if p.get("open_now") is not None else None,
                "distance": distance_str,
                "id": p.get("id"),
                "address": p.get("address"),
            }

            # Optional styling (map.js multiplies size/stroke by pinscale)
            if "color" in p:        pin["color"] = p["color"]
            if "size" in p:         pin["size"] = p["size"]
            if "stroke" in p:       pin["stroke"] = p["stroke"]
            if "strokeWidth" in p:  pin["strokeWidth"] = p["strokeWidth"]

            pins.append(pin)

        # ---- add "my location" if available --------------------------------------
        def _env_first(*names):
            for n in names:
                v = os.getenv(n)
                if v is not None and v != "":
                    return v
            return None

        if my_location is None:
            lat_s = _env_first("MY_LAT", "HOME_LAT", "HAL_HOME_LAT", "USER_LAT", "LAT")
            lon_s = _env_first("MY_LON", "HOME_LON", "HAL_HOME_LON", "USER_LON", "LON")
            try:
                if lat_s is not None and lon_s is not None:
                    my_location = (float(lat_s), float(lon_s))
            except Exception:
                my_location = None

        if my_location:
            my_lat, my_lon = my_location
            pins.append({
                "lat": float(my_lat),
                "lng": float(my_lon),
                "label": "You",
                "isUser": True,                  # <- used by map.js to style differently
                "color": "#2e9afe",              # blue fill
                "stroke": "#0b3d91",             # darker blue stroke
                "size": 8,                       # base radius; map.js scales by pinscale
                "strokeWidth": 3,
            })

        # Build URL for the vector map page
        url = (
            f"{self.base}/static/map.html?"
            f"key={quote(maptiler_key)}"
            f"&scale={s:.3g}"
            f"&uiscale={ui:.3g}"
            f"&pinscale={ps:.3g}"
            f"&zoom={int(zoom)}"
            f"&pins={quote(json.dumps(pins, separators=(',', ':')))}"
        )
        if center is not None and len(center) == 2:
            url += f"&lat={center[0]:.6f}&lng={center[1]:.6f}"

        return self.url(
            url,
            on=on,
            priority=priority,
            ttl=ttl,
            key=key,
            fullscreen=fullscreen,
            bg="#000",
        )

    def calendar(self,
                events: list[dict],
                *,
                on=("top",),
                priority: int = 80,
                ttl: int | None = 120,
                key: str | None = "calendar",
                fullscreen: bool = False,
                title: str = "Schedule",
                code: str = "GPM 72–KC",
                accent: str | None = None,
                limit: int | None = None):
        """
        Show a HAL-styled calendar board.

        events: list of dicts with fields like:
        { "title": str, "start": <iso or human>, "end": <iso or human>,
            "location": str, "open_now": bool }
        """
        q = f"title={quote(title)}&code={quote(code)}"
        if accent:
            q += f"&accent={quote(accent)}"
        if limit:
            q += f"&limit={int(limit)}"

        payload = quote(json.dumps(events, separators=(",", ":")))
        url = f"{self.base}/static/calendar.html?{q}&events={payload}"

        return self.url(
            url, on=on, priority=priority, ttl=ttl, key=key,
            fullscreen=fullscreen, bg="#000"
        )
