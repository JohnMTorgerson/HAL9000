"""Tap/hold controls and cancellation across real capture and voice-loop paths."""
import io
import threading
from unittest.mock import Mock, patch

import numpy as np
import pytest

from followup import ConversationCancelled, FollowupSession, FollowupSettings, FollowupDecision
from spacebar_trigger import SpacebarTrigger
from llm_client import LLMClient
import test_followup as followup_tests
import test_hal_recovery as hal_tests


@pytest.mark.parametrize('duration,cancel', [(.08, True), (.29, True), (.31, False), (2., False)])
def test_taps_are_consumed_and_holds_keep_the_original_press(duration, cancel):
    clock = hal_tests.SimulatedClock()
    tapped = Mock()
    keys = SpacebarTrigger(Mock(), on_tap=tapped, tap_enabled=lambda: True)
    with patch('spacebar_trigger.time', clock):
        keys.press()
        clock.advance(duration / 2)
        keys.press()  # Key repeat.
        clock.advance(duration / 2)
        keys.release()
    assert tapped.call_count == int(cancel)
    assert keys.snapshot() == ((None, None) if cancel else (0., duration))


def test_tap_eligibility_is_decided_at_press_and_listener_accepts_later_presses():
    clock = hal_tests.SimulatedClock()
    session = FollowupSession(FollowupSettings(True), clock.perf_counter)
    keys = SpacebarTrigger(Mock(), on_tap=session.cancel, tap_enabled=session.can_cancel)
    with patch('spacebar_trigger.time', clock):
        keys.press()  # Idle: a short explicit recording keeps its old behavior.
        session.begin_input()
        clock.advance(.1)
        keys.release()
        assert not session.cancelled.is_set()
        keys.prepare_capture()
        keys.press()  # Now processing a conversation: a tap ends it.
        clock.advance(.1)
        keys.release()
        assert session.cancelled.is_set()
        with pytest.raises(ConversationCancelled):
            session.after_response(explicit=True)
        session.finish_input()
        session.cancelled.clear()
        assert session.deadline() is None
        keys.press()
        clock.advance(.5)
        keys.release()
        assert keys.snapshot() == (.2, .7)
        assert not session.cancelled.is_set()


@pytest.mark.parametrize('held', [False, True])
def test_live_capture_tap_discards_audio_but_hold_overrides_silence_and_relevance(held):
    voice, detector, clock, state = followup_tests.CaptureTests().fixture(speech=(.2, .7))
    session = FollowupSession(FollowupSettings(True), clock.perf_counter)
    session.after_response(explicit=True)
    keys = SpacebarTrigger(Mock(), on_tap=session.cancel, tap_enabled=session.can_cancel)
    keys.available = True
    voice.keys = keys
    voice.check_cancelled = session.check_cancelled
    sleep = clock.sleep
    pressed = released = False

    def tick(seconds):
        nonlocal pressed, released
        sleep(seconds)
        if clock.now >= .65 and not pressed:
            keys.press()
            pressed = True
        if clock.now >= (1.5 if held else .75) and not released:
            keys.release()
            released = True
    clock.sleep = tick
    stream, triggered = Mock(), []
    with patch('voice_input.time', clock), patch('spacebar_trigger.time', clock):
        if held:
            audio, rate = voice.read_command(triggered.append, audio_stream=stream, followup_deadline=30)
            assert triggered == ['followup', 'spacebar']
            assert len(audio) > rate  # Kept recording beyond normal silence endpoint.
            assert not session.cancelled.is_set()
            stream.end_audio.assert_called_once()
        else:
            with pytest.raises(ConversationCancelled):
                voice.read_command(triggered.append, audio_stream=stream, followup_deadline=30)
            stream.cancel.assert_called_once()
            stream.end_audio.assert_not_called()
    assert state['capture_closed']
    assert not keys.stop_event.is_set()  # Shared listener survives microphone cleanup.


@pytest.mark.parametrize('stage', ['capture', 'transcription', 'decision', 'lookup', 'synthesis', 'playback'])
def test_tap_discards_inflight_followup_and_next_capture_requires_explicit_trigger(stage):
    ns, run, clock = followup_tests.MainLoopTests().fixture()
    reads = []
    ns['llm'].get_response.return_value = 'Initial answer.'
    ns['llm'].get_followup_response.return_value = FollowupDecision('respond', 'Unwanted answer.')

    def tap():
        controls = ns['voice_input'].start_controls.call_args.kwargs
        assert controls['tap_enabled']()
        controls['on_tap']()
        ns['sd'].stop.assert_called_once()
        ns['display_handler'].hide_conversation.assert_called_once()

    def read(on_trigger, **kwargs):
        reads.append(kwargs)
        if len(reads) == 1:
            on_trigger('spacebar')
        elif len(reads) == 2:
            assert kwargs['followup_deadline'] == clock.now + 30
            on_trigger('followup')
            if stage == 'capture':
                tap()
        else:
            assert 'followup_deadline' not in kwargs
            assert ns['llm'].finish_turn.call_count == 1
            assert ns['display_handler'].end_response.call_args.kwargs == {'followup_remaining': None}
            raise KeyboardInterrupt
        return [.1], 16000

    def transcribe(*args):
        if len(reads) == 2 and stage == 'transcription':
            tap()
        return 'A request.'

    def decision(*args, **kwargs):
        if stage == 'decision':
            tap()
        return FollowupDecision('respond', '[EXTERNAL_API_CALL] weather default' if stage == 'lookup' else 'Unwanted answer.')

    def lookup(*args, **kwargs):
        tap()
        return 'Lookup result arriving after cancellation.'

    def synthesize(*args, **kwargs):
        if len(reads) == 2 and stage == 'synthesis':
            tap()

    def play(*args, **kwargs):
        if len(reads) == 2 and stage == 'playback':
            tap()

    ns['voice_input'].read_command.side_effect = read
    ns['stt'].transcribe.side_effect = transcribe
    ns['llm'].get_followup_response.side_effect = decision
    ns['handle_api_call'].side_effect = lookup
    ns['voice'].synthesize_wav.side_effect = synthesize
    ns['play_audio'].side_effect = play
    with pytest.raises(SystemExit):
        run()
    ns['llm'].get_response.assert_called_once()  # No continuation after a cancelled lookup.
    assert ns['llm'].get_followup_response.call_count == int(stage not in ('capture', 'transcription'))
    assert ns['llm'].cancel_turn.call_count == int(stage not in ('capture', 'transcription'))
    ns['voice_input'].close_controls.assert_called_once()


def test_cancelled_client_history_drops_unspoken_reply_and_tool_steps_without_memory_write():
    client = LLMClient('ollama', 'fixture')
    original = [{'role': 'user', 'content': 'Accepted question.'},
                {'role': 'assistant', 'content': 'Spoken answer.'}]
    client.chat_history = original
    client.begin_turn('Background conversation.')
    client.chat_history = original + [{'role': 'assistant', 'content': 'An unspoken reply.'}]
    client.cancel_turn()
    assert client.chat_history == original


def test_tap_clears_output_while_network_call_is_still_blocked():
    ns, run, clock = followup_tests.MainLoopTests().fixture()
    waiting, release = threading.Event(), threading.Event()
    reads, failures = [], []
    ns['llm'].get_response.return_value = 'Initial answer.'

    def read(on_trigger, **kwargs):
        reads.append(kwargs)
        if len(reads) > 2:
            assert 'followup_deadline' not in kwargs
            raise KeyboardInterrupt
        on_trigger('spacebar' if len(reads) == 1 else 'followup')
        return [.1], 16000

    def slow_request(*args, **kwargs):
        waiting.set()
        assert release.wait(3)
        return FollowupDecision('respond', 'This late reply must be discarded.')

    def run_loop():
        try:
            run()
        except SystemExit:
            pass
        except BaseException as exc:
            failures.append(exc)

    ns['voice_input'].read_command.side_effect = read
    ns['llm'].get_followup_response.side_effect = slow_request
    thread = threading.Thread(target=run_loop)
    thread.start()
    try:
        assert waiting.wait(2)
        controls = ns['voice_input'].start_controls.call_args.kwargs
        assert controls['tap_enabled']()
        controls['on_tap']()
        assert thread.is_alive()  # Transport has not finished yet.
        ns['sd'].stop.assert_called_once()
        ns['display_handler'].hide_conversation.assert_called_once()
    finally:
        release.set()
        thread.join(3)
    assert not thread.is_alive() and not failures
    assert ns['llm'].finish_turn.call_count == 1
    assert ns['voice'].synthesize_wav.call_count == 1


def test_cancelled_image_search_cannot_publish_late_result():
    import test_image_lookup as image_tests
    session = FollowupSession(FollowupSettings(True))
    provider, display = Mock(supported=True), Mock()
    def lookup(query):
        session.cancel()
        return image_tests.ImageResult('ok', '', images=[image_tests.image()])
    provider.lookup.side_effect = lookup
    workflow = image_tests.ImageWorkflow(provider, display)
    with pytest.raises(ConversationCancelled):
        workflow.handle_reply(image_tests.COMMAND, check_cancelled=session.check_cancelled)
    display.present_image.assert_not_called()


@pytest.mark.parametrize('cancel_before_start', [False, True])
def test_actual_playback_checks_cancellation_before_start_and_after_wait(cancel_before_start):
    clock = hal_tests.SimulatedClock()
    session = FollowupSession(FollowupSettings(True), clock.perf_counter)
    sd, sf, logger = Mock(), Mock(), Mock()
    sf.read.return_value = (np.ones((160, 2), dtype=np.float32), 16000)
    namespace = dict(time=clock, logger=logger, sd=sd, np=np, io=io, sf=sf,
                     get_default_device=lambda _: (None, 16000))
    play = hal_tests.load_hal_function('play_audio', namespace)
    if cancel_before_start:
        session.cancel()
    else:
        sd.wait.side_effect = session.cancel
    with pytest.raises(ConversationCancelled):
        play('fixture.wav', preserve_mastering=True, conversation=session)
    assert sd.play.call_count == int(not cancel_before_start)
