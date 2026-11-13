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


def test_message_levels_survive_trimming_and_remain_plain_text(delivery):
    d = delivery
    d.handler.min_push_interval = 0
    messages = [(25, 'HAL: I found an error in that story.'),
                (30, 'Warning line one.\nWarning line two.'),
                (40, 'Error line.'), (50, 'Critical line.')]
    for level, text in messages:
        d.handler.handle(logging.LogRecord('HAL', level, '', 0, text, (), None))
    sent = d.client.text.call_args
    assert [line['level'] for line in sent.kwargs['text_lines']] == [
        'conversation', 'warning', 'error', 'error']
    plain = '\n'.join(text for _, text in messages)
    assert sent.args[0] == plain
    # Cut inside the warning, not at a message boundary.
    d.handler.max_chars = len(plain) - plain.index('Warning line two.')
    lines = d.handler._compose_lines()
    assert ''.join(line['text'] for line in lines) == '…\n' + plain[-d.handler.max_chars:]
    assert lines[1] == {'text': 'Warning line two.', 'level': 'warning'}
    assert '\x1b' not in d.handler._compose_text()


def test_raw_http_fallback_carries_the_same_level_metadata(delivery):
    d = delivery
    d.handler._client = None
    d.handler._session = Mock()
    d.handler.handle(logging.LogRecord('HAL', 40, '', 0, 'Error message.', (), None))
    payload = d.handler._session.post.call_args.kwargs['json']
    assert payload['text'] == 'Error message.'
    assert payload['text_lines'] == [{'text': 'Error message.', 'level': 'error'}]


@pytest.mark.parametrize('window', [8, 30, 60, 180])
def test_followup_owns_transcript_deadline_and_diagnostics_cannot_renew_it(delivery, window):
    d = delivery
    d.handler.follow_conversation()
    d.handler.begin_capture()
    d.say('USER: Hello.')
    d.handler.begin_response()
    d.say('HAL: Hello.')
    d.handler.end_response(followup_remaining=window)
    assert d.client.text.call_args.kwargs['ttl'] == window
    d.clock.now += 3
    d.handler.handle(logging.LogRecord('HAL', 30, '', 0, 'Background warning.', (), None))
    assert d.client.text.call_args.kwargs['ttl'] == window - 3
    assert d.client.text.call_args.kwargs['text_lines'][-1]['level'] == 'warning'
    d.handler.begin_capture()  # Candidate speech holds the pane through processing.
    assert d.client.text.call_args.kwargs['ttl'] is None
    d.clock.now += 2
    d.handler.end_response(followup_remaining=window - 5)  # Ignored: same deadline.
    assert d.client.text.call_args.kwargs['ttl'] == window - 5
    d.handler.end_response(followup_remaining=None)  # Timeout or explicit end.
    d.client.clear.assert_called_with(key='logs')
    pushes = d.client.text.call_count
    d.handler.handle(logging.LogRecord('HAL', 40, '', 0, 'Idle error.', (), None))
    d.say('A stale reply must not reopen the pane.')
    assert d.client.text.call_count == pushes


def test_cancel_replaces_pending_text_retry_with_clear_and_fresh_conversation(delivery):
    d = delivery
    d.handler.follow_conversation()
    d.clock.now += 1
    d.client.text.side_effect = OSError('offline')
    d.handler.begin_response()
    d.say('HAL: Old reply.')
    stale_timer = d.timers[-1]
    d.handler.hide_conversation()
    stale_timer.fire()
    d.timers[-1].fire()
    assert d.client.text.call_count == 1
    assert d.client.clear.call_count == 2  # Startup and cancellation.
    d.client.text.side_effect = None
    d.handler.begin_capture()
    d.say('USER: A new conversation.')
    if d.timers[-1] is not stale_timer:
        d.timers[-1].fire()
    assert 'Old reply' not in d.client.text.call_args.args[0]
    assert d.client.text.call_args.kwargs['ttl'] is None


def test_raw_http_clear_only_removes_the_transcript_overlay(delivery):
    d = delivery
    d.handler._client = None
    d.handler._session = Mock()
    d.handler.follow_conversation()
    assert d.handler._session.post.call_args.args[0].endswith('/api/clear')
    assert d.handler._session.post.call_args.kwargs['json'] == {'key': 'logs'}


def test_long_processing_keeps_pane_alive_until_its_actual_window_starts(delivery):
    d = delivery
    d.handler.follow_conversation()
    d.clock.now += 1
    d.handler.begin_response()
    d.say('HAL: Just a moment.')
    for _ in range(7):
        d.timers[-1].fire()
        assert d.client.text.call_args.kwargs['ttl'] is None
    assert d.clock.now > 300  # Beyond the display server's 180-second idle reset.
    heartbeat = d.timers[-1]
    d.handler.end_response(followup_remaining=60)
    sent = d.client.text.call_count
    heartbeat.fire()  # A stale heartbeat cannot undo the deadline.
    assert d.client.text.call_count == sent
    assert d.client.text.call_args.kwargs['ttl'] == 60
