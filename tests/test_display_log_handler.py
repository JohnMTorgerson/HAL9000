"""Transcript delivery under burst logging, failed HTTP pushes and long speech."""
import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import display_log_handler as module


@pytest.fixture
def delivery(monkeypatch):
    clock = SimpleNamespace(now=100.)
    timers = []

    class Timer:
        def __init__(self, delay, callback):
            self.at, self.callback, self.cancelled = clock.now + delay, callback, False

        def start(self):
            timers.append(self)

        def cancel(self):
            self.cancelled = True

        def fire(self):
            clock.now = max(clock.now, self.at)
            self.callback()

    client = Mock()
    monkeypatch.setattr(module, 'DisplayClient', Mock(return_value=client))
    monkeypatch.setattr(module, 'Timer', Timer)
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=lambda: clock.now))
    handler = module.DisplayPushHandler()
    handler.setFormatter(logging.Formatter('%(message)s'))

    def say(text):
        handler.handle(logging.LogRecord('HAL', 25, '', 0, text, (), None))

    yield SimpleNamespace(handler=handler, client=client, clock=clock, timers=timers, say=say)
    handler.close()


def test_burst_is_delivered_without_waiting_for_another_log_record(delivery):
    d = delivery
    d.say('USER: Try again?')
    d.clock.now += .005
    d.say('HAL: Just a moment.')
    assert d.client.text.call_count == 1
    d.timers[-1].fire()
    assert d.client.text.call_count == 2
    assert d.client.text.call_args.args[0] == 'USER: Try again?\nHAL: Just a moment.'
    assert d.client.text.call_args.kwargs['ttl'] == 30


def test_response_holds_text_then_starts_full_ttl_after_playback(delivery):
    d = delivery
    d.handler.begin_response()
    d.say('HAL: A long response.')
    assert d.client.text.call_args.kwargs['ttl'] is None
    d.clock.now += 45
    d.handler.end_response()
    assert d.client.text.call_args.kwargs['ttl'] == 30
    assert d.client.text.call_args.args[0] == 'HAL: A long response.'
    d.handler.end_response()  # Ignored/empty follow-up must not refresh it.
    assert d.client.text.call_count == 2


def test_transient_failure_retries_latest_text_without_a_new_record(delivery, caplog):
    d = delivery
    d.client.text.side_effect = [OSError('fixture failure'), None]
    with caplog.at_level('DEBUG', logger='HAL'):
        d.say('USER: Hello.')
        d.clock.now += 1
        d.say('HAL: Hello.')
        old = d.timers[0]
        old.fire()  # A cancelled callback racing with replacement is harmless.
        assert d.client.text.call_count == 1
        d.timers[-1].fire()
    assert d.client.text.call_count == 2
    assert d.client.text.call_args.args[0] == 'USER: Hello.\nHAL: Hello.'
    assert d.client.text.call_args.kwargs['ttl'] == 26
    assert 'delivery failed' in caplog.text and 'delivery recovered' in caplog.text
    assert all(r.skip_display for r in caplog.records)


@pytest.mark.parametrize('close', [False, True])
def test_pending_retry_cannot_resurrect_expired_or_closed_transcript(delivery, close):
    d = delivery
    d.client.text.side_effect = OSError('fixture failure')
    d.say('HAL: Old text.')
    if close:
        d.handler.close()
    else:
        d.clock.now += 31
    d.timers[-1].fire()
    assert d.client.text.call_count == 1


def test_delivery_diagnostics_do_not_become_transcript_lines(delivery):
    d = delivery
    record = logging.LogRecord('HAL', 30, '', 0, 'Delivery failed.', (), None)
    record.skip_display = True
    d.handler.handle(record)
    d.client.text.assert_not_called()
