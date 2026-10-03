"""Song commands and real HAL control flow, without a microphone or paid calls."""
import io
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import resample_poly

from followup import FollowupDecision, FollowupSettings
from llm_client import LLMClient
from song_request import (DAISY_PATH, SONG_FAILURE_REPLY, SONG_PAUSE_SECONDS,
                          SongRequestError, parse_song_request)
import test_hal_recovery as hal_tests


COMMAND = '[PLAY_SONG] {"song": "daisy", "intro": "I\'d be happy to."}'


def test_parse_song_command_and_leave_ordinary_speech_alone():
    assert parse_song_request('  ' + COMMAND + '\n').intro == "I'd be happy to."
    for text in ('Daisy Bell was written in 1892.', 'I can sing Daisy Bell.',
                 '[EXTERNAL_API_CALL] weather default'):
        assert parse_song_request(text) is None


@pytest.mark.parametrize('payload', [
    '', 'not JSON', 'null', '[]', '{}',
    '{"song":"../../other.wav","intro":"Certainly."}',
    '{"song":"daisy","intro":"Certainly.","path":"other.wav"}',
    '{"song":"daisy","intro":null}',
    '{"song":"daisy","intro":"  "}',
    '{"song":"daisy","intro":"[EXTERNAL_API_CALL] weather default"}',
    '{"song":"daisy","intro":"Certainly.\\nSinging now."}',
    json.dumps({'song': 'daisy', 'intro': 'x' * 201}),
    '{"song":"daisy","intro":"Certainly."} extra prose',
])
def test_reject_malformed_commands_instead_of_speaking_them(payload):
    with pytest.raises(SongRequestError):
        parse_song_request('[PLAY_SONG] ' + payload)


def loop_fixture():
    ns, run = hal_tests.MainLoopTests().fixture()
    clock = hal_tests.SimulatedClock()
    clock.sleep = lambda seconds: clock.advance(seconds)
    ns.update(time=clock, wave=MagicMock(), voice=Mock(), syn_config=object(),
              USER='fixture', strip_name_at_sentence_end=lambda text, name: text,
              sf=Mock(), followup_settings=FollowupSettings(enabled=True))
    ns['sf'].read.return_value = ([.1], 16000)
    return ns, run, clock


@pytest.mark.parametrize('followup', [False, True])
def test_intro_pause_song_and_history_finish_before_microphone_reopens(followup):
    ns, run, clock = loop_fixture()
    events, reads = [], []
    ns['stt'].transcribe.side_effect = ['Hello.', 'Sing it again.'] if followup else ['Sing a song.']
    ns['llm'].get_response.return_value = 'Hello.' if followup else COMMAND
    ns['llm'].get_followup_response.return_value = FollowupDecision('respond', COMMAND)

    def speak(text, *args, **kwargs):
        events.append(('speak', text, clock.now))
    ns['voice'].synthesize_wav.side_effect = speak

    def play(filename, **kwargs):
        events.append(('play', kwargs['label'], clock.now))
        if kwargs['label'].startswith('song'):
            assert filename == str(DAISY_PATH)
            assert kwargs['preserve_mastering'] is True
            assert not kwargs.get('first_response', False)
            clock.advance(50.)
            assert ns['voice_input'].read_command.call_count == (2 if followup else 1)
        else:
            clock.advance(2.)
        events.append(('finished', kwargs['label'], clock.now))
    ns['play_audio'].side_effect = play

    def finish(user, reply, **kwargs):
        events.append(('history', reply, clock.now))
        if 'action_result' in kwargs:
            assert kwargs['action_result'] == 'Played the Daisy Bell recording to completion.'
            assert events[-2][0:2] == ('finished', 'song: Daisy Bell')
    ns['llm'].finish_turn.side_effect = finish

    def read(on_trigger, **kwargs):
        reads.append(kwargs)
        if len(reads) == 1:
            on_trigger('wakeword')
        elif followup and len(reads) == 2:
            on_trigger('followup')
        else:
            assert events[-1][0] == 'history'
            assert kwargs['followup_deadline'] == clock.now + 8
            ns['DisplayServerManager'].return_value.stop.assert_not_called()
            raise KeyboardInterrupt
        return [.1], 16000
    ns['voice_input'].read_command.side_effect = read

    # Check intro-end -> song-start is exactly the configured silent pause.
    def play_with_order(filename, **kwargs):
        if kwargs['label'].startswith('song'):
            assert events[-1][0:2] == ('finished', 'reply')
            assert clock.now - events[-1][2] == SONG_PAUSE_SECONDS
        play(filename, **kwargs)
    ns['play_audio'].side_effect = play_with_order
    with pytest.raises(SystemExit):
        run()
    assert events[-1][1] == "I'd be happy to."
    assert [call.kwargs['label'] for call in ns['play_audio'].call_args_list] == (
        ['reply', 'reply', 'song: Daisy Bell'] if followup else ['reply', 'song: Daisy Bell'])
    ns['handle_api_call'].assert_not_called()
    assert ns['llm'].get_response.call_count == 1
    assert ns['llm'].get_followup_response.call_count == int(followup)


@pytest.mark.parametrize('failure', ['malformed', 'missing', 'playback'])
def test_song_failures_report_failure_and_resume_listening(failure, tmp_path):
    ns, run, clock = loop_fixture()
    ns['llm'].get_response.return_value = '[PLAY_SONG] invalid' if failure == 'malformed' else COMMAND
    if failure == 'missing':
        ns['DAISY_PATH'] = tmp_path / 'not-present.wav'
    def play(filename, **kwargs):
        if kwargs['label'].startswith('song'):
            raise OSError('fixture output-device failure')
    ns['play_audio'].side_effect = play
    def read(on_trigger, **kwargs):
        if ns['voice_input'].read_command.call_count == 1:
            on_trigger('spacebar')
            return [.1], 16000
        ns['DisplayServerManager'].return_value.stop.assert_not_called()
        result = ns['llm'].finish_turn.call_args.kwargs['action_result']
        assert 'failed' in result
        assert 'to completion' not in result
        raise KeyboardInterrupt
    ns['voice_input'].read_command.side_effect = read
    with pytest.raises(SystemExit):
        run()
    assert ns['llm'].finish_turn.call_count == 1
    spoken = ns['voice'].synthesize_wav.call_args.args[0]
    assert '[PLAY_SONG]' not in spoken
    if failure != 'playback':
        assert spoken == SONG_FAILURE_REPLY
        assert ns['play_audio'].call_count == 1
    else:
        ns['logger'].display.assert_any_call(f'HAL: {SONG_FAILURE_REPLY}')


@pytest.mark.parametrize('persistent', [False, True])
@pytest.mark.parametrize('result', [
    'Played the Daisy Bell recording to completion.',
    'Daisy Bell playback failed; completion was not confirmed.',
])
def test_history_and_optional_memory_record_actual_result(persistent, result):
    memory = Mock() if persistent else None
    if memory is not None:
        memory.read_context.return_value = None
    client = LLMClient('ollama', 'fixture', memory=memory)
    client.chat_history = [{'role': 'user', 'content': 'Sing.'},
                           {'role': 'assistant', 'content': COMMAND}]
    client.finish_turn('Sing.', 'Certainly.', action_result=result)
    expected = f'Certainly.\n[Application action result: {result}]'
    assert client.chat_history[-1] == {'role': 'assistant', 'content': expected}
    assert '[PLAY_SONG]' not in json.dumps(client.chat_history)
    if memory is not None:
        memory.record_turn.assert_called_once_with('Sing.', expected, client.turn_started_at)


@pytest.mark.parametrize('device_rate', [48000, 44100])
def test_mastered_recording_preserves_level_and_bypasses_voice_eq(device_rate):
    data, rate = sf.read(DAISY_PATH, dtype='float32', always_2d=True)
    assert rate == 44100
    assert len(data) / rate > 40
    sd = Mock()
    ns = {'time': SimpleNamespace(perf_counter=lambda: 0.), 'logger': Mock(),
          'sd': sd, 'np': np, 'io': io, 'sf': sf, 'resample_poly': resample_poly,
          'HI_PASS_FREQ': 200, 'AudioSegment': Mock(),
          'get_default_device': lambda kind: ('fixture output', device_rate)}
    play = hal_tests.load_hal_function('play_audio', ns)
    play(str(DAISY_PATH), preserve_mastering=True)
    ns['AudioSegment'].from_file.assert_not_called()
    played = sd.play.call_args.args[0]
    expected = np.tile(data, (1, 2)) if data.shape[1] == 1 else data
    if device_rate != rate:
        divisor = np.gcd(device_rate, rate)
        expected = resample_poly(expected, device_rate // divisor, rate // divisor, axis=0)
    np.testing.assert_array_equal(played, expected)
    assert sd.play.call_args.kwargs == {'samplerate': device_rate, 'device': 'fixture output'}
    sd.wait.assert_called_once()
