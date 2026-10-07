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
import base64
import binascii
import hashlib
import io
from itertools import count
import logging
import random
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Set, Literal

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder  # <-- Option A: encode Pydantic -> JSON-able
from pydantic import BaseModel, Field, field_validator
from image_files import MAX_IMAGE_BYTES, web_url

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
class Citation(BaseModel):
    url: str
    label: str = Field(max_length=200)

    @field_validator('url')
    @classmethod
    def validate_url(cls, value):
        if not web_url(value):
            raise ValueError('Citation must be an HTTP(S) webpage URL')
        return value


class Panel(BaseModel):
    type: ContentType                 # "image", "text", or "url"
    src: Optional[str] = None         # image path or URL, or iframe URL when type="url"
    text: Optional[str] = None        # used when type="text"
    fit: Literal["cover", "contain"] = "cover"  # image object-fit behavior
    bg: str = "#000"                  # background color (e.g., "#000" black)
    citations: List[Citation] = Field(default_factory=list, max_length=6)
    load_token: Optional[str] = None


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
logger = logging.getLogger('HAL')
_server_id = uuid.uuid4().hex
_render_revisions = count(1)


# ---------------------------------------------
# Slideshow + idle handling
# ---------------------------------------------
# How often each pane advances (seconds); pane changes are staggered by half.
ROTATE_SECONDS = 120

# Lists of images for top/bottom panels. Change to your filenames.
ROTATION = {
    "top":    [f"{SCREENS_DIR}/screen_06.png", f"{SCREENS_DIR}/screen_04.png", f"{SCREENS_DIR}/screen_05.png", f"{SCREENS_DIR}/screen_03.png"],
    "bottom": [f"{SCREENS_DIR}/screen_01.png", f"{SCREENS_DIR}/screen_02.png", f"{SCREENS_DIR}/screen_08.png", f"{SCREENS_DIR}/screen_07.png", f"{SCREENS_DIR}/screen_00.png"],
}

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
    # Order snapshots across HTTP and WebSocket delivery. A delayed message must
    # not roll the browser back to old text or resurrect an expired overlay.
    version = {'server_id': _server_id, 'revision': next(_render_revisions)}
    # Filter out expired overlays (render-path safety; cleanup also happens in a background task)
    valid = [o for o in overlays if (o.expires_at is None or o.expires_at > now())]

    # Fullscreen overlay? Highest priority wins.
    fs = [o for o in valid if o.fullscreen]
    if fs:
        topdog = sorted(fs, key=lambda o: o.priority, reverse=True)[0]
        return {**version, "layout": "fullscreen", "top": topdog.panel, "bottom": topdog.panel}

    # Otherwise "split" layout: pick best per slot or use slideshow 'state'
    render_top = state["top"]
    render_bot = state["bottom"]

    for slot in ("top", "bottom"):
        # Candidates that explicitly target this slot
        cands = [o for o in valid if slot in o.slots]
        if cands:
            # Most recently pushed overlay wins ties (including image -> map).
            winner = max(enumerate(cands), key=lambda pair: (pair[1].priority, pair[0]))[1]
            if slot == "top":
                render_top = winner.panel
            else:
                render_bot = winner.panel

    # Each panel owns its citations; an image never keeps the log pane open.
    return {**version, "layout": "split", "top": render_top, "bottom": render_bot}


async def broadcast(msg: dict) -> None:
    """
    Send a JSON message to all connected WebSocket clients.
    Drops any clients that have gone away.
    """
    # Option A boundary: convert Pydantic models into pure JSON-able types
    payload = jsonable_encoder(msg, exclude_none=True)
    dead: List[WebSocket] = []
    for ws in tuple(clients):
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
async def get_state() -> dict:
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
    citations: List[Citation] = Field(default_factory=list, max_length=6)
    load_token: Optional[str] = None

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
                o.panel = Panel(type=req.type, src=req.src, text=req.text, fit=req.fit, bg=req.bg,
                                citations=req.citations, load_token=req.load_token)
                o.priority = req.priority
                o.fullscreen = req.fullscreen
                o.slots = {"top", "bottom"} if req.fullscreen else set(req.slots or ["top"])
                o.expires_at = (now() + req.ttl_secs) if req.ttl_secs else None
                overlays.remove(o)
                overlays.append(o)
                await broadcast_render()
                return {"ok": True, "id": o.id}

    # Create a new overlay
    slots = {"top", "bottom"} if req.fullscreen else set(req.slots or ["top"])
    overlay = Overlay(
        slots=slots,
        panel=Panel(type=req.type, src=req.src, text=req.text, fit=req.fit, bg=req.bg,
                    citations=req.citations, load_token=req.load_token),
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


# A bounded, same-origin image cache also works when the display is on another
# machine. HAL uploads only the image it is about to show, not remote URLs.
image_loads: Dict[str, str] = {}
image_visibility: Dict[str, str] = {}


class ImageDisplayRequest(BaseModel):
    data: str = Field(max_length=12 * 1024 * 1024)
    citations: List[Citation] = Field(min_length=1, max_length=6)
    ttl_secs: int = Field(default=120, ge=10, le=600)


class ImageLoadRequest(BaseModel):
    token: str = Field(min_length=32, max_length=32)
    status: Literal['received', 'loaded', 'error']
    visibility: Literal['visible', 'hidden'] | None = None


@app.post('/api/images/show')
async def show_lookup_image(req: ImageDisplayRequest):
    try:
        from PIL import Image
        data = base64.b64decode(req.data, validate=True)
        if not data or len(data) > MAX_IMAGE_BYTES:
            raise ValueError('Invalid image size')
        with Image.open(io.BytesIO(data)) as im:
            if im.format != 'JPEG' or not (64 <= min(im.size) <= max(im.size) <= 1600):
                raise ValueError('Expected a normalized JPEG')
            im.verify()
    except (ValueError, OSError, binascii.Error) as exc:
        raise HTTPException(400, 'Invalid image data') from exc
    except ImportError as exc:
        raise HTTPException(503, 'Install requirements-images.txt on the display server') from exc
    cache = BASE / 'media' / 'image-search'
    cache.mkdir(parents=True, exist_ok=True)
    filename = hashlib.sha256(data).hexdigest() + '.jpg'
    path = cache / filename
    path.write_bytes(data)
    # Keep at most twenty images and remove files older than one day.
    for index, old in enumerate(sorted(cache.glob('*.jpg'), key=lambda p: p.stat().st_mtime, reverse=True)):
        if old != path and (index >= 20 or time.time() - old.stat().st_mtime > 86400):
            old.unlink(missing_ok=True)
    token = uuid.uuid4().hex
    image_loads.clear()
    image_visibility.clear()
    image_loads[token] = 'pending'
    await push_overlay(PushRequest(type='image', src='/media/image-search/' + filename,
                                   slots=['top'], fit='cover', key='image-lookup', priority=80,
                                   ttl_secs=req.ttl_secs, citations=req.citations, load_token=token))
    return {'ok': True, 'token': token}


@app.post('/api/images/loaded')
async def image_loaded(req: ImageLoadRequest):
    render = compute_render()
    if (render['layout'] == 'split' and render['top'].load_token == req.token
            and req.token in image_loads):
        # One successful browser is enough; a broken secondary browser must not
        # overwrite confirmation from the working kiosk.
        if (image_loads[req.token] != 'loaded'
                and (req.status != 'received' or image_loads[req.token] == 'pending')):
            image_loads[req.token] = req.status
            if req.visibility is not None:
                image_visibility[req.token] = req.visibility
        return {'ok': True}
    return {'ok': False}


@app.get('/api/images/status/{token}')
def image_status(token: str):
    render = compute_render()
    status = image_loads.get(token, 'pending')
    if render['layout'] != 'split' or render['top'].load_token != token:
        status = 'hidden'
    return {'status': status, 'connected_browsers': len(clients),
            'document_visibility': image_visibility.get(token)}


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
    logger.info('Display browser connected: connections=%d', len(clients))
    try:
        # Include the initial send in cleanup if a browser disconnects at once.
        initial = {"type": "render", "payload": compute_render()}
        await ws.send_json(jsonable_encoder(initial, exclude_none=True))
        while True:
            # We don't need client -> server messages yet; this keeps the socket alive.
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(ws)
        logger.info('Display browser disconnected: connections=%d', len(clients))


# ---------------------------------------------
# Background tasks:
# 1) rotator: periodically advance the slideshow images
# 2) reaper: expire overlays and perform idle reset
# ---------------------------------------------
def shuffled_images(images):
    """Visit each distinct image once per shuffle, avoiding boundary repeats."""
    images = list(dict.fromkeys(images))
    previous = None
    while images:
        cycle = images.copy()
        random.shuffle(cycle)
        if len(cycle) > 1 and cycle[0] == previous:
            other = random.randrange(1, len(cycle))
            cycle[0], cycle[other] = cycle[other], cycle[0]
        for image in cycle:
            previous = image
            yield image


async def rotator() -> None:
    """
    Independently shuffle each pane, starting with a random pair immediately.
    Each pane advances every ROTATE_SECONDS, staggered by half that interval.
    Overlays (if any) still win on top of this base state.
    """
    cycles = {slot: shuffled_images(ROTATION.get(slot) or []) for slot in ("top", "bottom")}

    def advance(slot):
        image = next(cycles[slot], None)
        if image is not None:
            state[slot].type = "image"
            state[slot].src = image

    for slot in cycles:
        advance(slot)
    await broadcast_render()

    while True:
        for slot in ("bottom", "top"):
            await asyncio.sleep(ROTATE_SECONDS / 2)
            advance(slot)
            # Push new render (overlays may still be in effect).
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
class QuietPollingAccessFilter(logging.Filter):
    """Omit successful polling requests, retaining HTTP errors and other logs."""
    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING or not isinstance(record.args, tuple) or len(record.args) != 5:
            return True
        _, method, path, _, status = record.args
        if not isinstance(path, str) or not isinstance(status, int):
            return True
        path = path.split('?', 1)[0]
        polling = path == '/api/state' or path.startswith('/api/images/status/')
        return not (method == 'GET' and polling and 200 <= status < 400)


@app.on_event("startup")
async def on_startup() -> None:
    # Install after Uvicorn configures logging, for embedded and standalone use.
    access_logger = logging.getLogger('uvicorn.access')
    if not any(isinstance(f, QuietPollingAccessFilter) for f in access_logger.filters):
        access_logger.addFilter(QuietPollingAccessFilter())
    asyncio.create_task(rotator())
    asyncio.create_task(reaper())
