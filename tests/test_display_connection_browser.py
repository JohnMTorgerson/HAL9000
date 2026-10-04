"""Exercise display recovery with an actual browser and HTTP display server."""
from concurrent.futures import ThreadPoolExecutor
import io
import json
import logging
import os
import socket
import threading
import time
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(os.getenv('HAL_BROWSER_TESTS') != '1',
                                reason='Opt-in test requires Playwright and Chromium')


@pytest.fixture
def display_browser():
    from PIL import Image
    from playwright.sync_api import sync_playwright
    import uvicorn
    from display import display_server as srv
    from display_client import DisplayClient
    from display_log_handler import DisplayPushHandler

    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    servers = []

    def start():
        srv.overlays.clear()
        srv.clients.clear()
        srv.image_loads.clear()
        srv.image_visibility.clear()
        server = uvicorn.Server(uvicorn.Config(srv.app, host='127.0.0.1', port=port, log_level='error'))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        servers.append((server, thread))
        deadline = time.monotonic() + 5
        while not server.started and time.monotonic() < deadline:
            time.sleep(.02)
        assert server.started

    def stop():
        server, thread = servers[-1]
        server.should_exit = True
        thread.join(5)
        assert not thread.is_alive()

    def restart():
        stop()
        start()

    start()
    display = DisplayClient(f'http://127.0.0.1:{port}')
    display._s.trust_env = False
    handler = DisplayPushHandler(display.base, min_push_interval=0)
    handler._client._s.trust_env = False
    raw = io.BytesIO()
    Image.new('RGB', (600, 300), 'blue').save(raw, format='JPEG')
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True, args=['--no-sandbox'])
            page = browser.new_page(viewport={'width': 900, 'height': 1200})
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))

            def say(text):
                handler.handle(logging.LogRecord('HAL', 25, '', 0, text, (), None))
                # Prove the text reached the server before testing browser delivery.
                assert text in display._s.get(display.base + '/api/state').json()['bottom']['text']

            def present():
                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(display.present_image, raw.getvalue(), citations=[{
                        'url': 'https://example.com/plane', 'label': 'Source'}])
                    while not future.done():
                        page.wait_for_timeout(20)
                    assert future.result() == 'loaded'

            try:
                yield SimpleNamespace(page=page, display=display, srv=srv, say=say,
                                      present=present, restart=restart)
                assert not errors
            finally:
                browser.close()
    finally:
        stop()
        display._s.close()
        handler._client._s.close()
        handler.close()
        srv.overlays.clear()
        srv.clients.clear()
        srv.image_loads.clear()
        srv.image_visibility.clear()


@pytest.mark.parametrize('socket_state', [0, 1], ids=['stuck_connecting', 'silently_open'])
def test_first_text_and_image_arrive_even_without_a_working_websocket(display_browser, socket_state):
    fixture = display_browser
    page = fixture.page
    # Model a handshake that never completes or a stale OPEN socket. Neither
    # delivers onclose; the server really has zero WebSocket clients throughout.
    page.add_init_script("""(() => {
        window.stalledSockets = [];
        window.WebSocket = class {
            static CONNECTING = 0; static OPEN = 1; static CLOSED = 3;
            constructor() { this.readyState = SOCKET_STATE; window.stalledSockets.push(this); }
            close() { this.readyState = 2; }
        };
    })();""".replace('SOCKET_STATE', str(socket_state)))
    page.goto(fixture.display.base)
    page.wait_for_selector('#bottom img')
    fixture.say('USER: Show me a picture of an X-29.')
    page.wait_for_function("document.querySelector('#bottom .text-content')?.textContent.includes('X-29')",
                           timeout=3500)
    fixture.present()
    assert not fixture.srv.clients
    assert page.locator('#top .citations a').inner_text() == 'example.com'
    fixture.say('HAL: Here it is.')
    page.wait_for_function("document.querySelector('#bottom .text-content')?.textContent.includes('Here it is')",
                           timeout=3500)
    if socket_state == 0:
        assert page.evaluate('stalledSockets.length') >= 2
        # A late frame from an abandoned connection must not restore stale text.
        page.evaluate("""stalledSockets[0].onmessage({data: JSON.stringify({type: 'render', payload: {
            layout: 'split', top: {type: 'text', text: 'stale'},
            bottom: {type: 'text', text: 'stale'}
        }})})""")
        assert 'Here it is' in page.locator('#bottom .text-content').inner_text()


def test_blocked_initial_state_does_not_block_websocket_or_restart_recovery(display_browser):
    fixture = display_browser
    page = fixture.page
    # Leave HTTP snapshots pending: their timeout must free the next poll, and
    # the socket must connect independently of the initial fetch.
    page.add_init_script("""(() => {
        const realFetch = window.fetch.bind(window);
        window.stateFetchAttempts = 0;
        window.fetch = (url, options) => {
            if (url !== '/api/state') return realFetch(url, options);
            window.stateFetchAttempts += 1;
            return new Promise((resolve, reject) => {
                options?.signal?.addEventListener('abort', () =>
                    reject(new DOMException('Request timed out', 'AbortError')));
            });
        };
    })();""")
    page.goto(fixture.display.base)
    page.wait_for_selector('#bottom img', timeout=3000)
    fixture.say('USER: The initial state request is stuck.')
    page.wait_for_function("document.querySelector('#bottom .text-content')?.textContent.includes('stuck')",
                           timeout=3000)
    fixture.present()
    # Keep the same browser page while HAL's display server stops and restarts.
    fixture.restart()
    fixture.say('USER: First request after restarting HAL.')
    fixture.present()
    page.wait_for_function("document.querySelector('#bottom .text-content')?.textContent.includes('restarting')",
                           timeout=3000)
    page.wait_for_function('stateFetchAttempts >= 2', timeout=4000)


def test_delayed_http_snapshot_cannot_overwrite_a_newer_socket_update(display_browser):
    fixture = display_browser
    page = fixture.page
    pending = []

    def delay_snapshot(route):
        # Capture the old body now, but deliver it after a newer socket update.
        response = route.fetch()
        pending.append((route, response.body()))

    page.goto(fixture.display.base)
    page.wait_for_selector('#bottom img')
    fixture.say('USER: Old conversation text.')
    page.wait_for_function("document.querySelector('#bottom .text-content')?.textContent.includes('Old')")
    page.route('**/api/state', delay_snapshot)
    page.evaluate("window.dispatchEvent(new Event('online'))")
    deadline = time.monotonic() + 2
    while not pending and time.monotonic() < deadline:
        page.wait_for_timeout(10)
    assert pending
    route, old_body = pending[0]
    assert 'Old conversation' in json.loads(old_body)['bottom']['text']
    fixture.say('HAL: New conversation text.')
    page.wait_for_function("document.querySelector('#bottom .text-content')?.textContent.includes('New')")
    with page.expect_response('**/api/state') as response:
        route.fulfill(status=200, content_type='application/json', body=old_body)
    response.value.finished()
    page.wait_for_timeout(100)
    assert 'New conversation' in page.locator('#bottom .text-content').inner_text()
