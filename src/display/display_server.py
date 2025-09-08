"""
display_server.py (v2, heavily commented)

A tiny web server that renders a two-panel "dashboard" in a browser (kiosk).
- Default content is a slideshow (top + bottom panels).
- You can "push" time-limited overlays to either panel or to fullscreen.
- Overlays are prioritized: higher priority wins when multiple things target the same panel.
- After inactivity or when overlays expire, the screen reverts to the slideshow.
- Updates are pushed to the browser via WebSocket in real time.

Why this architecture?
- It’s future-proof (images, text, pages, maps, logs… all just "panel content").
- It’s deterministic (priority + TTL), avoids last-write-wins race conditions.
- It’s decoupled: HAL (or any module) only makes HTTP calls; the display decides what to show.
"""

import asyncio
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Set, Literal

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder  # <-- Option A: encode Pydantic -> JSON-able
from pydantic import BaseModel, Field

# ---------------------------------------------
# Type aliases for clarity (useful in signatures)
# ---------------------------------------------
PanelSlot = Literal["top", "bottom"]         # which half of the vertical screen
LayoutMode = Literal["split", "fullscreen"]  # split = two panels; fullscreen = one content on both
ContentType = Literal["image", "text", "url"]  # how the frontend should render the content


# ---------------------------------------------
# FastAPI app + static mounts
# ---------------------------------------------
app = FastAPI()
BASE = Path(__file__).parent

# /static: serve your index.html, CSS, JS, etc.
# /media:  serve your local images (slideshow assets)
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
app.mount("/media", StaticFiles(directory=BASE / "media"), name="media")

# slideshow images directory
SCREENS_DIR = "/media/screens"

# ---------------------------------------------
# Panel model: describes WHAT to render in a slot
# The browser decides HOW to draw it (img/iframe/text)
# ---------------------------------------------
class Panel(BaseModel):
    type: ContentType                 # "image", "text", or "url"
    src: Optional[str] = None         # image path or URL, or iframe URL when type="url"
    text: Optional[str] = None        # used when type="text"
    fit: Literal["cover", "contain"] = "cover"  # image object-fit behavior
    bg: str = "#000"                  # background color (e.g., "#000" black)


# ---------------------------------------------
# Default "base" state (what slideshow updates)
# This is the fallback when no overlays are present.
# ---------------------------------------------
state: Dict[PanelSlot, Panel] = {
    "top":    Panel(type="image", src=f"{SCREENS_DIR}/screen_06.png", fit="cover", bg="#000"),
    "bottom": Panel(type="image", src=f"{SCREENS_DIR}/screen_01.png", fit="cover", bg="#000"),
}


# ---------------------------------------------
# Overlay model: a time-limited, prioritized piece
# of content that can target one or both panels.
# ---------------------------------------------
class Overlay(BaseModel):
    # Unique ID (auto-generated) so clients can clear by ID if needed
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)

    # Set of target slots: {"top"}, {"bottom"}, or {"top","bottom"}
    # If "fullscreen" is True, this is ignored (applies to both)
    slots: Set[PanelSlot] = Field(default_factory=lambda: {"top"})

    # What to render (Panel above)
    panel: Panel

    # Priority: higher wins when multiple overlays target the same slot
    # Suggested ranges:
    #  - slideshow baseline: 10
    #  - generic push: 50
    #  - logs: 70
    #  - main results (maps, charts): 80
    #  - fullscreen takeover: 100
    #  - critical alert: 120
    priority: int = 50

    # When does this overlay expire? (epoch seconds). None means "until cleared"
    expires_at: Optional[float] = None

    # Fullscreen override (shows this panel on BOTH top and bottom, regardless of slots)
    fullscreen: bool = False

    # Optional "key" to upsert the same logical overlay (e.g., "logs", "map")
    # Using a key avoids duplicate overlays and lets you refresh TTL easily.
    key: Optional[str] = None


# Active overlays (sorted / filtered at render time)
overlays: List[Overlay] = []

# Connected WebSocket clients (browsers)
clients: Set[WebSocket] = set()


# ---------------------------------------------
# Slideshow + idle handling
# ---------------------------------------------
# How often to advance the slideshow images (seconds)
ROTATE_SECONDS = 120

# Lists of images for top/bottom panels. Change to your filenames.
ROTATION = {
    "top":    [f"{SCREENS_DIR}/screen_06.png", f"{SCREENS_DIR}/screen_04.png", f"{SCREENS_DIR}/screen_05.png", f"{SCREENS_DIR}/screen_06.png", f"{SCREENS_DIR}/screen_03.png"],
    "bottom": [f"{SCREENS_DIR}/screen_01.png", f"{SCREENS_DIR}/screen_02.png", f"{SCREENS_DIR}/screen_08.png", f"{SCREENS_DIR}/screen_07.png", f"{SCREENS_DIR}/screen_00.png"],
}

# Current slideshow index into the ROTATION lists
_slideshow_idx = 0

# Idle reset: if no pushes happen for this many seconds, clear overlays (belt + suspenders)
IDLE_RESET_SECS = 180
_last_activity_ts = time.time()


# ---------------------------------------------
# Utility helpers
# ---------------------------------------------
def now() -> float:
    """Current time in epoch seconds."""
    return time.time()


def touch_activity() -> None:
    """Mark 'activity' so idle reset timer is postponed."""
    global _last_activity_ts
    _last_activity_ts = now()


def prune_expired() -> bool:
    """
    Remove overlays whose expires_at has passed.
    Returns True if the overlay list changed.
    """
    t = now()
    before = len(overlays)
    overlays[:] = [o for o in overlays if (o.expires_at is None or o.expires_at > t)]
    return before != len(overlays)


def compute_render() -> dict:
    """
    Compute the effective "render" payload for the browser:
      {
        "layout": "split" | "fullscreen",
        "top": Panel,
        "bottom": Panel
      }

    Rules:
    - If any unexpired fullscreen overlay exists, the single highest-priority
      one wins and is used for BOTH panels.
    - Otherwise, for each panel, pick the highest-priority unexpired overlay
      that targets that panel. If none, fall back to the slideshow 'state'.
    """
    # Filter out expired overlays (render-path safety; cleanup also happens in a background task)
    valid = [o for o in overlays if (o.expires_at is None or o.expires_at > now())]

    # Fullscreen overlay? Highest priority wins.
    fs = [o for o in valid if o.fullscreen]
    if fs:
        topdog = sorted(fs, key=lambda o: o.priority, reverse=True)[0]
        return {"layout": "fullscreen", "top": topdog.panel, "bottom": topdog.panel}

    # Otherwise "split" layout: pick best per slot or use slideshow 'state'
    render_top = state["top"]
    render_bot = state["bottom"]

    for slot in ("top", "bottom"):
        # Candidates that explicitly target this slot
        cands = [o for o in valid if slot in o.slots]
        if cands:
            winner = sorted(cands, key=lambda o: o.priority, reverse=True)[0]
            if slot == "top":
                render_top = winner.panel
            else:
                render_bot = winner.panel

    return {"layout": "split", "top": render_top, "bottom": render_bot}


async def broadcast(msg: dict) -> None:
    """
    Send a JSON message to all connected WebSocket clients.
    Drops any clients that have gone away.
    """
    # Option A boundary: convert Pydantic models into pure JSON-able types
    payload = jsonable_encoder(msg, exclude_none=True)
    dead: List[WebSocket] = []
    for ws in clients:
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.discard(ws)


async def broadcast_render() -> None:
    """Compute current render state and push it to all clients."""
    await broadcast({"type": "render", "payload": compute_render()})


# ---------------------------------------------
# HTTP routes
# ---------------------------------------------
@app.get("/")
def index() -> HTMLResponse:
    """
    Serve the main dashboard page.
    The file static/index.html should include the two-panel layout and
    a WebSocket client that listens for {"type":"render"} messages.
    """
    return HTMLResponse((BASE / "static" / "index.html").read_text(encoding="utf-8"))


@app.get("/api/state")
def get_state() -> dict:
    """
    Return the current computed render state (useful for initial load or debugging).
    """
    # Option A boundary: make sure nested Panels are encoded as JSON-able dicts
    return jsonable_encoder(compute_render(), exclude_none=True)


# ---------------------------------------------
# Push API: upsert a new overlay (or refresh by key)
# ---------------------------------------------
class PushRequest(BaseModel):
    # What to show
    type: ContentType
    src: Optional[str] = None
    text: Optional[str] = None
    fit: Literal["cover", "contain"] = "cover"
    bg: str = "#000"

    # Where/how to show it
    slots: Optional[List[PanelSlot]] = None  # default ["top"] unless fullscreen=True
    priority: int = 50
    ttl_secs: Optional[int] = 120           # how long it should stay (None = until cleared)
    fullscreen: bool = False
    key: Optional[str] = None               # logical key for "upsert" behavior

@app.post("/api/push")
async def push_overlay(req: PushRequest) -> dict:
    """
    Push an overlay onto the display:
      - If 'key' is provided and matches an existing overlay, update that overlay in place
        (panel, priority, slots/fullscreen, expiration).
      - Otherwise, create a new overlay.

    This endpoint is idempotent with respect to 'key': reusing the same key lets you
    refresh TTL or swap content without multiplying overlays.
    """
    touch_activity()

    # Upsert by key (update existing overlay)
    if req.key:
        for o in overlays:
            if o.key == req.key:
                # Update properties in place
                o.panel = Panel(type=req.type, src=req.src, text=req.text, fit=req.fit, bg=req.bg)
                o.priority = req.priority
                o.fullscreen = req.fullscreen
                o.slots = {"top", "bottom"} if req.fullscreen else set(req.slots or ["top"])
                o.expires_at = (now() + req.ttl_secs) if req.ttl_secs else None
                await broadcast_render()
                return {"ok": True, "id": o.id}

    # Create a new overlay
    slots = {"top", "bottom"} if req.fullscreen else set(req.slots or ["top"])
    overlay = Overlay(
        slots=slots,
        panel=Panel(type=req.type, src=req.src, text=req.text, fit=req.fit, bg=req.bg),
        priority=req.priority,
        expires_at=(now() + req.ttl_secs) if req.ttl_secs else None,
        fullscreen=req.fullscreen,
        key=req.key,
    )
    overlays.append(overlay)
    await broadcast_render()
    return {"ok": True, "id": overlay.id}


# ---------------------------------------------
# Clear API: remove overlays by id, key, slot, or all
# ---------------------------------------------
class ClearRequest(BaseModel):
    slot: Optional[Literal["top", "bottom", "both"]] = None  # clear overlays targeting this slot
    key: Optional[str] = None                                 # clear overlay(s) that match this key
    ids: Optional[List[str]] = None                           # clear by explicit overlay ids
    all: bool = False                                         # nuke everything

@app.post("/api/clear")
async def clear_overlay(req: ClearRequest) -> dict:
    """
    Clear overlays selectively:
      - {"all": true}                                 -> clear every overlay
      - {"ids": ["abc", "def"]}                       -> clear specific overlays by ID
      - {"key": "logs"}                               -> clear overlay(s) by logical key
      - {"slot": "bottom"} or {"slot": "both"}        -> remove overlays that target those slots
                                                         (fullscreen overlays also cleared when slot=both)
    """
    touch_activity()
    changed = False

    if req.all:
        overlays.clear()
        changed = True

    elif req.ids:
        target = set(req.ids)
        before = len(overlays)
        overlays[:] = [o for o in overlays if o.id not in target]
        changed = len(overlays) != before

    elif req.key:
        before = len(overlays)
        overlays[:] = [o for o in overlays if o.key != req.key]
        changed = len(overlays) != before

    elif req.slot:
        # Convert "both" to {"top","bottom"}; otherwise single slot set
        wanted = {"top", "bottom"} if req.slot == "both" else {req.slot}
        before = len(overlays)
        # Keep overlays whose slots do NOT intersect the requested target set.
        # Note: we also clear fullscreen overlays when slot == both (since they cover both)
        overlays[:] = [o for o in overlays if (o.fullscreen and req.slot != "both") or o.slots.isdisjoint(wanted)]
        changed = len(overlays) != before

    if changed:
        await broadcast_render()
    return {"ok": True, "changed": changed}


# ---------------------------------------------
# WebSocket: push render changes in real time
# ---------------------------------------------
@app.websocket("/ws")
async def ws(ws: WebSocket) -> None:
    """
    Each browser connects to /ws to receive real-time updates.
    On connect:
      - accept the socket
      - add to clients
      - push the current render state immediately
    Then keep the socket open. We ignore inbound messages for now.
    """
    await ws.accept()
    clients.add(ws)
    # Send initial state so the page renders immediately
    initial = {"type": "render", "payload": compute_render()}
    await ws.send_json(jsonable_encoder(initial, exclude_none=True))  # <-- Option A boundary
    try:
        while True:
            # We don't need client -> server messages yet; this keeps the socket alive.
            await ws.receive_text()
    except WebSocketDisconnect:
        clients.discard(ws)


# ---------------------------------------------
# Background tasks:
# 1) rotator: periodically advance the slideshow images
# 2) reaper: expire overlays and perform idle reset
# ---------------------------------------------
async def rotator() -> None:
    """
    Every ROTATE_SECONDS, advance the slideshow 'state'.
    Overlays (if any) still win on top of this base state.
    """
    global _slideshow_idx
    while True:
        await asyncio.sleep(ROTATE_SECONDS)
        _slideshow_idx += 1

        # Update slideshow images for each slot (wrap around lists)
        for slot in ("top", "bottom"):
            imgs = ROTATION.get(slot) or []
            if imgs:
                state[slot].type = "image"
                state[slot].src = imgs[_slideshow_idx % len(imgs)]

        # Push new render (overlays may still be in effect)
        await broadcast_render()


async def reaper() -> None:
    """
    Once per second:
      - Drop expired overlays (based on expires_at).
      - If there has been no "activity" (push/clear) for IDLE_RESET_SECS,
        clear all overlays (belt + suspenders) so the slideshow returns.
    """
    while True:
        await asyncio.sleep(1)
        changed = prune_expired()

        # Idle reset: ensure we eventually revert even if an overlay was pushed with no TTL
        if now() - _last_activity_ts > IDLE_RESET_SECS:
            if overlays:
                overlays.clear()
                changed = True

        if changed:
            await broadcast_render()


# ---------------------------------------------
# Startup: kick off background tasks
# ---------------------------------------------
@app.on_event("startup")
async def on_startup() -> None:
    asyncio.create_task(rotator())
    asyncio.create_task(reaper())
