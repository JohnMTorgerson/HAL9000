"""The embedded server logs to the terminal without closing HAL's handlers."""
import io
import logging
import socket
from unittest.mock import Mock

import pytest
import requests

from display import display_server as srv
from display.server_lifecycle import DisplayServerManager
from display_log_handler import DisplayPushHandler


@pytest.fixture
def isolated_server_logging(monkeypatch):
    for name in ('uvicorn', 'uvicorn.error', 'uvicorn.access', 'uvicorn.asgi'):
        target = logging.getLogger(name)
        for attr, value in (('handlers', []), ('filters', []), ('level', logging.NOTSET),
                            ('propagate', True), ('disabled', False)):
            monkeypatch.setattr(target, attr, value)


def test_embedded_server_logs_survive_restart_and_leave_transcripts_working(
        isolated_server_logging, monkeypatch, capsys):
    # Other display tests suppress lifespan tasks; this test exercises real startup.
    monkeypatch.setattr(srv.app.router, 'on_startup', [srv.on_startup])
    monkeypatch.setattr(srv.app.router, 'on_shutdown', [])
    srv.overlays.clear()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    server = DisplayServerManager(url, logger=Mock(), register_atexit=False)
    # This must exist BEFORE Uvicorn starts to reproduce the old dictConfig bug.
    transcript = DisplayPushHandler(url, min_push_interval=0)
    transcript._client._s.trust_env = False
    session = requests.Session()
    session.trust_env = False
    try:
        for attempt in range(2):
            server.start()
            assert server._server.started
            assert not transcript._stopped
            message = f'Transcript after server start {attempt}.'
            transcript.handle(logging.LogRecord('HAL', 25, '', 0, message, (), None))
            state = session.get(url + '/api/state', timeout=2)
            assert state.ok and message in state.json()['bottom']['text']
            session.get(url + '/api/state?poll=1', timeout=2).raise_for_status()
            assert session.get(url + '/missing-route', timeout=2).status_code == 404
            # Errors on a normally silent polling endpoint must still be visible.
            logging.getLogger('uvicorn.access').info('%s - "%s %s HTTP/%s" %d',
                '127.0.0.1:1234', 'GET', '/api/state', '1.1', 503)
            logging.getLogger('uvicorn.error').warning('Server warning remains visible.')
            logging.getLogger('uvicorn.error').error('Server error remains visible.')
            thread = server._thread
            server.stop()
            assert not thread.is_alive()
            assert not transcript._stopped
            out = capsys.readouterr()
            assert out.err.count('Application startup complete.') == 1
            assert out.err.count('Application shutdown complete.') == 1
            assert 'Server warning remains visible.' in out.err
            assert 'Server error remains visible.' in out.err
            assert out.out.count('POST /api/push HTTP/1.1') == 1
            assert 'GET /missing-route HTTP/1.1" 404' in out.out
            assert 'GET /api/state HTTP/1.1" 503' in out.out
            assert 'GET /api/state HTTP/1.1" 200' not in out.out
            assert '/api/state?poll=1' not in out.out
    finally:
        server.stop()
        session.close()
        transcript.close()
        transcript._client._s.close()
        srv.overlays.clear()


def test_existing_server_handlers_are_preserved(isolated_server_logging):
    stream = io.StringIO()
    existing = logging.StreamHandler(stream)
    target = logging.getLogger('uvicorn.access')
    target.addHandler(existing)
    server = DisplayServerManager(register_atexit=False)
    try:
        server._configure_terminal_logging()
        server._configure_terminal_logging()
        assert target.handlers == [existing]
        assert len(logging.getLogger('uvicorn').handlers) == 1
        server.stop()
        assert target.handlers == [existing]
        assert not existing._closed
    finally:
        server.stop()
        target.removeHandler(existing)
        existing.close()
