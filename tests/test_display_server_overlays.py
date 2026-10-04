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
    srv.image_loads.clear()
    srv.image_visibility.clear()

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


def test_image_citation_stays_in_top_panel_while_logs_expire_independently(monkeypatch, client):
    clock = {'now': 1000}
    monkeypatch.setattr(srv, 'now', lambda: clock['now'])
    citation = {'url': 'https://example.com/product', 'label': 'Example product'}
    push(client, type='image', src='/media/test.jpg', key='image-lookup',
         citations=[citation], slots=['top'], priority=80, ttl_secs=120)
    push(client, type='text', text='HAL: Here it is.', key='logs', slots=['bottom'],
         priority=70, ttl_secs=30)
    state = get_render(client)
    assert state['top']['type'] == 'image'
    assert state['bottom']['text'] == 'HAL: Here it is.'
    assert state['top']['citations'] == [citation]
    assert not state['bottom']['citations']
    push(client, type='text', text='USER: Another one.', key='logs', slots=['bottom'], ttl_secs=30)
    assert get_render(client)['top']['citations'] == [citation]
    clock['now'] = 1031
    state = get_render(client)
    assert state['top']['citations'] == [citation]
    assert state['bottom']['type'] == 'image' and not state['bottom']['citations']
    clock['now'] = 1121
    assert not get_render(client)['top']['citations']


def test_citation_updates_on_next_and_disappears_on_close_or_other_content(client):
    for number in (1, 2):
        citation = {'url': f'https://example.com/{number}', 'label': f'Image {number}'}
        push(client, type='image', src=f'/media/{number}.jpg', key='image-lookup',
             citations=[citation], slots=['top'], priority=80)
        assert get_render(client)['top']['citations'] == [citation]
    push(client, type='url', src='/static/map.html', key='map', slots=['top'], priority=80)
    assert not get_render(client)['top']['citations']
    clear(client, key='map')
    assert get_render(client)['top']['citations']
    clear(client, key='image-lookup')
    assert not get_render(client)['top']['citations']


def test_upload_and_browser_acknowledgment_requires_current_visible_token(client, tmp_path, monkeypatch):
    import base64
    import io
    from PIL import Image
    monkeypatch.setattr(srv, 'BASE', tmp_path)
    raw = io.BytesIO()
    Image.new('RGB', (200, 100), 'red').save(raw, format='JPEG')
    payload = {'data': base64.b64encode(raw.getvalue()).decode(),
               'citations': [{'url': 'https://example.com/source', 'label': 'Source'}]}
    first = client.post('/api/images/show', json=payload)
    assert first.status_code == 200
    token = first.json()['token']
    assert client.get(f'/api/images/status/{token}').json() == {
        'status': 'pending', 'connected_browsers': 0, 'document_visibility': None}
    with client.websocket_connect('/ws') as ws:
        ws.receive_json()
        assert client.get(f'/api/images/status/{token}').json()['connected_browsers'] == 1
        assert client.post('/api/images/loaded', json={
            'token': token, 'status': 'received', 'visibility': 'hidden'}).json()['ok']
        assert client.get(f'/api/images/status/{token}').json() == {
            'status': 'received', 'connected_browsers': 1, 'document_visibility': 'hidden'}
    assert client.post('/api/images/loaded', json={'token': token, 'status': 'loaded'}).json()['ok']
    assert client.get(f'/api/images/status/{token}').json()['status'] == 'loaded'
    # A late "received" report or a secondary browser's error cannot undo success.
    for status in ('received', 'error'):
        client.post('/api/images/loaded', json={'token': token, 'status': status})
        assert client.get(f'/api/images/status/{token}').json()['status'] == 'loaded'
    # Re-showing the same cached file still needs a fresh acknowledgment.
    second = client.post('/api/images/show', json=payload).json()['token']
    assert second != token
    assert not client.post('/api/images/loaded', json={'token': token, 'status': 'loaded'}).json()['ok']
    assert client.get(f'/api/images/status/{second}').json()['status'] == 'pending'
    assert get_render(client)['top']['fit'] == 'cover'
    assert len(list((tmp_path / 'media' / 'image-search').glob('*.jpg'))) == 1
    clear(client, key='image-lookup')
    assert client.get(f'/api/images/status/{second}').json()['status'] == 'hidden'


@pytest.mark.parametrize('url', ['javascript:alert(1)', 'file:///secret', 'https://user:pass@example.com'])
def test_citation_links_reject_active_schemes_and_credentials(client, url):
    assert client.post('/api/push', json={'type': 'image', 'src': '/media/test.jpg',
        'citations': [{'url': url, 'label': 'test'}]}).status_code == 422


def test_invalid_upload_does_not_replace_current_image(client):
    push(client, type='image', src='/media/good.jpg', key='image-lookup')
    response = client.post('/api/images/show', json={
        'data': 'not base64', 'citations': [{'url': 'https://example.com', 'label': 'Source'}]})
    assert response.status_code == 400
    assert get_render(client)['top']['src'] == '/media/good.jpg'
