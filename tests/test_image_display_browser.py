"""Optional real-browser coverage: HAL_BROWSER_TESTS=1 pytest this_file.py."""
import io
import os
import socket
import threading
import time

import pytest

pytestmark = pytest.mark.skipif(os.getenv('HAL_BROWSER_TESTS') != '1',
                                reason='Opt-in test requires Playwright and Chromium')


def test_image_load_and_pinned_clickable_citation_in_real_browser():
    from PIL import Image, ImageDraw
    from playwright.sync_api import sync_playwright
    import uvicorn
    from display import display_server as srv
    from display_client import DisplayClient

    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    srv.overlays.clear()
    srv.image_loads.clear()
    server = uvicorn.Server(uvicorn.Config(srv.app, host='127.0.0.1', port=port, log_level='error'))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not server.started and time.monotonic() < deadline:
        time.sleep(.02)
    assert server.started
    display = DisplayClient(f'http://127.0.0.1:{port}')
    display._s.trust_env = False
    drawing = Image.new('RGB', (1200, 700), '#172838')
    painter = ImageDraw.Draw(drawing)
    painter.rounded_rectangle((350, 50, 850, 650), radius=50, fill='#b5d6eb')
    painter.rounded_rectangle((375, 100, 825, 590), radius=20, fill='#23405a')
    raw = io.BytesIO()
    drawing.save(raw, format='JPEG')
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=['--no-sandbox'])
            try:
                page = browser.new_page(viewport={'width': 900, 'height': 1200})
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.goto(display.base)
                page.wait_for_selector('#top img')
                display.text('\n'.join(f'Conversation line {n}' for n in range(30)),
                             on=('bottom',), priority=70, key='logs')
                citations = [{'url': 'https://www.example.com/phone?model=1', 'label': 'Example product page'}]
                assert display.present_image(raw.getvalue(), citations=citations) == 'loaded'
                link = page.locator('#top .citations a')
                assert link.inner_text() == 'example.com'
                assert link.get_attribute('href') == citations[0]['url']
                assert link.get_attribute('target') == '_blank'
                assert page.locator('#bottom .citations').count() == 0
                box, upper = link.bounding_box(), page.locator('#top').bounding_box()
                assert box['y'] >= upper['y']
                assert 0 < upper['y'] + upper['height'] - box['y'] - box['height'] < 40
                assert 0 < upper['x'] + upper['width'] - box['x'] - box['width'] < 40
                assert page.locator('#top img').evaluate("el => getComputedStyle(el).objectFit") == 'cover'
                display.text('HAL: Here it is.', on=('bottom',), priority=70, key='logs')
                page.wait_for_function("document.querySelector('#bottom .text-content').textContent === 'HAL: Here it is.'")
                assert link.inner_text() == 'example.com'
                if os.getenv('HAL_BROWSER_SCREENSHOT'):
                    page.screenshot(path=os.environ['HAL_BROWSER_SCREENSHOT'])
                # Expiring the conversation cannot strand a credit in the bottom pane.
                display.clear(key='logs')
                page.wait_for_selector('#bottom img')
                assert page.locator('#bottom .citations').count() == 0
                assert link.inner_text() == 'example.com'
                # Reusing the same picture produces a fresh load acknowledgment
                # and refreshes its source; stale acknowledgments cannot win.
                assert display.present_image(raw.getvalue(), citations=[{
                    'url': 'https://example.com/other', 'label': '<img src=x onerror=alert(1)>'}]) == 'loaded'
                assert page.locator('#top .citations img').count() == 0
                assert link.inner_text() == 'example.com'
                display.clear(key='image-lookup')
                page.wait_for_function("document.querySelectorAll('.citations a').length === 0")
                assert not errors
            finally:
                browser.close()
    finally:
        server.should_exit = True
        thread.join(5)
        display._s.close()
        srv.overlays.clear()
        srv.clients.clear()
        srv.image_loads.clear()
