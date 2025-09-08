# tests/test_display_server_overlays.py
import pytest
from fastapi.testclient import TestClient

# IMPORTANT: import from the package path you set up (pythonpath=src)
from display import display_server as srv


@pytest.fixture(autouse=True)
def disable_lifespan_and_reset_state():
    """
    Disable FastAPI startup/shutdown so background tasks never start,
    and reset all mutable globals so each test is deterministic.
    """
    # Disable startup/shutdown handlers entirely
    srv.app.router.on_startup.clear()
    srv.app.router.on_shutdown.clear()

    # Reset overlays & clients
    srv.overlays.clear()
    srv.clients.clear()

    # Reset slideshow base state
    srv.state["top"] = srv.Panel(type="image", src=f"{srv.SCREENS_DIR}/screen_06.png", fit="cover", bg="#000")
    srv.state["bottom"] = srv.Panel(type="image", src=f"{srv.SCREENS_DIR}/screen_01.png", fit="cover", bg="#000")

    # Reset timers/counters
    srv._slideshow_idx = 0
    srv._last_activity_ts = srv.now()


@pytest.fixture
def client():
    # Plain TestClient is fine now that lifespan hooks are cleared
    with TestClient(srv.app) as c:
        yield c


def get_render(client):
    r = client.get("/api/state")
    assert r.status_code == 200
    return r.json()


def push(client, **kwargs):
    payload = {
        "type": "text",
        "text": "payload",
        "slots": ["top"],
        "priority": 50,
        "ttl_secs": None,
        "fullscreen": False,
        "key": None,
        "fit": "cover",
        "bg": "#000",
        "src": None,
    }
    payload.update(kwargs)
    r = client.post("/api/push", json=payload)
    assert r.status_code == 200
    return r.json()


def clear(client, **kwargs):
    r = client.post("/api/clear", json=kwargs)
    assert r.status_code == 200
    return r.json()


# -------------------
# Core behavior tests
# -------------------

def test_initial_state_split_layout(client):
    s = get_render(client)
    assert s["layout"] == "split"
    assert s["top"]["type"] == "image"
    assert s["bottom"]["type"] == "image"
    assert s["top"]["src"].startswith(f"{srv.SCREENS_DIR}/")
    assert s["bottom"]["src"].startswith(f"{srv.SCREENS_DIR}/")


def test_push_top_overlay_replaces_top_only(client):
    push(client, type="text", text="Hello Top", slots=["top"])
    s = get_render(client)
    assert s["layout"] == "split"
    assert s["top"]["type"] == "text" and s["top"]["text"] == "Hello Top"
    assert s["bottom"]["type"] == "image"


def test_priority_wins_latest_does_not_override_lower_priority(client):
    push(client, type="text", text="P50", slots=["top"], priority=50)
    push(client, type="text", text="P80", slots=["top"], priority=80)
    s = get_render(client)
    assert s["top"]["text"] == "P80"
    push(client, type="text", text="P40", slots=["top"], priority=40)
    s = get_render(client)
    assert s["top"]["text"] == "P80"


def test_fullscreen_overlay_overrides_both_panels(client):
    push(client, type="text", text="Bottom Only", slots=["bottom"], priority=70)
    push(client, type="text", text="FULL", fullscreen=True, priority=100)
    s = get_render(client)
    assert s["layout"] == "fullscreen"
    assert s["top"]["text"] == "FULL"
    assert s["bottom"]["text"] == "FULL"


def test_upsert_by_key_updates_in_place(client):
    a = push(client, type="text", text="logs v1", slots=["bottom"], priority=70, key="logs", ttl_secs=300)
    id1 = a["id"]
    b = push(client, type="text", text="logs v2", slots=["bottom"], priority=80, key="logs", ttl_secs=300)
    id2 = b["id"]
    assert id1 == id2
    assert len([o for o in srv.overlays if o.key == "logs"]) == 1
    s = get_render(client)
    assert s["bottom"]["type"] == "text" and s["bottom"]["text"] == "logs v2"


def test_clear_by_key_slot_and_ids(client):
    a = push(client, type="text", text="A", slots=["top"], priority=60, key="A")
    b = push(client, type="text", text="B", slots=["bottom"], priority=60, key="B")
    clear(client, key="A")
    s = get_render(client)
    assert s["top"]["type"] == "image"
    assert s["bottom"]["type"] == "text" and s["bottom"]["text"] == "B"
    clear(client, slot="bottom")
    s = get_render(client)
    assert s["bottom"]["type"] == "image"
    a2 = push(client, type="text", text="ID1", slots=["top"], priority=60)
    b2 = push(client, type="text", text="ID2", slots=["bottom"], priority=60)
    clear(client, ids=[a2["id"]])
    s = get_render(client)
    assert s["top"]["type"] == "image"
    assert s["bottom"]["type"] == "text" and s["bottom"]["text"] == "ID2"


def test_clear_all_nukes_everything(client):
    push(client, type="text", text="A", slots=["top"])
    push(client, type="text", text="B", slots=["bottom"])
    clear(client, all=True)
    s = get_render(client)
    assert s["top"]["type"] == "image"
    assert s["bottom"]["type"] == "image"
    assert srv.overlays == []


def test_ttl_expiry_via_prune_expired(monkeypatch, client):
    t0 = 1_000_000.0
    fake_time = {"t": t0}
    monkeypatch.setattr(srv, "now", lambda: fake_time["t"])

    push(client, type="text", text="TTL10", slots=["top"], ttl_secs=10, priority=80)
    s = get_render(client)
    assert s["top"]["type"] == "text" and s["top"]["text"] == "TTL10"

    fake_time["t"] = t0 + 11
    assert srv.prune_expired() is True
    s = get_render(client)
    assert s["top"]["type"] == "image"


def test_priority_per_slot_is_independent(client):
    push(client, type="text", text="Top-50", slots=["top"], priority=50)
    push(client, type="text", text="Top-80", slots=["top"], priority=80)
    push(client, type="text", text="Bot-60", slots=["bottom"], priority=60)
    push(client, type="text", text="Bot-55", slots=["bottom"], priority=55)
    s = get_render(client)
    assert s["layout"] == "split"
    assert s["top"]["text"] == "Top-80"
    assert s["bottom"]["text"] == "Bot-60"


def test_fullscreen_beats_any_slot_overlays(client):
    push(client, type="text", text="Top-99", slots=["top"], priority=99)
    push(client, type="text", text="Bot-99", slots=["bottom"], priority=99)
    push(client, type="text", text="FS-100", fullscreen=True, priority=100)
    s = get_render(client)
    assert s["layout"] == "fullscreen"
    assert s["top"]["text"] == "FS-100"
    assert s["bottom"]["text"] == "FS-100"
