"""Device selection and API failure recovery without accounts or native audio."""
import ast
import copy
import json
from pathlib import Path
import re
import shlex
import sys
import types
import unittest
from unittest.mock import MagicMock, Mock, patch

import httpx
import numpy as np
from openai import OpenAI

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))
from audio_devices import choose_input_device
from audio_capture import AudioOverflowError
from llm_client import LLMClient, LLMServiceError
from live_transcription import TranscriptionError
from followup import FollowupSettings, FollowupSession, explicitly_addresses_hal


def load_hal_function(name, namespace):
    # HAL currently initializes accounts, audio models and GPIO at import time.
    # Execute its actual function body without triggering those startup effects.
    source = ast.parse((SRC / 'hal.py').read_text())
    function = next(node for node in source.body
                    if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, str(SRC / 'hal.py'), 'exec'), namespace)
    return namespace[name]


def device(name, inputs=1, rate=48000):
    return {'name': name, 'max_input_channels': inputs,
            'max_output_channels': 2, 'default_samplerate': rate}


class DeviceTests(unittest.TestCase):
    def test_mac_virtual_default_uses_fifine_at_its_native_rate(self):
        devices = [device('Virtual Desktop Mic', rate=44100),
                   device('USB speakers', inputs=0),
                   device('fifine Microphone', rate=48000)]
        sd = types.SimpleNamespace(query_devices=lambda: devices,
                                   default=types.SimpleNamespace(device=(0, 1)))
        select = load_hal_function('get_default_device', {
            'sd': sd, 'SYSTEM': 'Darwin', 'choose_input_device': choose_input_device})
        self.assertEqual(select('input'), (2, 48000))
        self.assertEqual(select('output'), (None, 44100))

    def test_usb_input_is_preferred_on_mac_and_pi(self):
        devices = [device('Built-in Microphone'), device('USB Audio Device')]
        self.assertEqual(choose_input_device(devices, default_input=0), 1)

    def test_physical_default_is_kept_without_fifine_or_usb(self):
        devices = [device('Microphone One'), device('Microphone Two')]
        self.assertEqual(choose_input_device(devices, default_input=1), 1)

    def test_virtual_names_cannot_win_usb_preference(self):
        for name in ('USB Virtual Mic', 'BlackHole 2ch', 'Soundflower (2ch)',
                     'Loopback Audio', 'Aggregate Device', 'Monitor of USB Audio',
                     'CABLE Output (VB-Audio Virtual Cable)'):
            with self.subTest(name=name):
                devices = [device(name), device('Built-in Microphone')]
                self.assertEqual(choose_input_device(devices, default_input=0), 1)

    def test_no_physical_input_requires_an_explicit_choice(self):
        for devices in ([], [device('Speaker', inputs=0)], [device('Virtual Desktop Mic')]):
            with self.subTest(devices=devices):
                with self.assertRaisesRegex(RuntimeError, 'HAL_INPUT_DEVICE'):
                    choose_input_device(devices, default_input=0)


def error_response(status, code, error_type='server_error', param=None):
    return httpx.Response(status, json={'error': {
        'message': 'This raw server message must not become a HAL reply.',
        'type': error_type, 'param': param, 'code': code}})


def successful_response(service_tier=None):
    body = {
        'id': 'chatcmpl-test', 'object': 'chat.completion', 'created': 0,
        'model': 'test-model', 'choices': [{'index': 0, 'finish_reason': 'stop',
            'message': {'role': 'assistant', 'content': 'I am ready.'}}]}
    if service_tier is not None:
        body['service_tier'] = service_tier
    return httpx.Response(200, json=body)


class LLMTests(unittest.TestCase):
    def make_client(self, handler, model_name='test-model', **options):
        http_client = httpx.Client(transport=httpx.MockTransport(handler))
        self.addCleanup(http_client.close)
        with patch('llm_client.OpenAI', side_effect=lambda **kwargs:
                   OpenAI(http_client=http_client, **kwargs)):
            client = LLMClient('openai', model_name, max_history=2,
                               openai_api_key='test-placeholder-not-a-real-key', **options)
        self.addCleanup(client.client.close)
        return client

    def test_fast_applies_to_initial_and_external_followup_with_reasoning_disabled(self):
        requests = []
        def handler(request):
            requests.append(json.loads(request.content))
            return successful_response('priority' if len(requests) == 1 else 'fast')
        client = self.make_client(handler, model_name='gpt-6-luna', service_tier='fast')
        with self.assertLogs('HAL', level='INFO') as logs:
            client.get_response('What is next on my calendar?')
            client.get_response('[EXTERNAL_API_RESPONSE] {"title": "fixture event"}')
        self.assertEqual(len(requests), 2)
        for request in requests:
            self.assertEqual(request['service_tier'], 'priority')
            self.assertEqual(request['model'], 'gpt-6-luna')
            self.assertEqual(request['reasoning_effort'], 'none')
            self.assertEqual(request['max_completion_tokens'], 512)
        self.assertIn('requested=fast; used=priority', logs.output[0])
        self.assertIn('requested=fast; used=fast', logs.output[1])

    def test_tier_can_be_omitted_or_explicitly_set_to_standard(self):
        for setting, expected in [(None, None), ('', None), ('auto', 'auto'),
                                  ('default', 'default'), ('priority', 'priority'),
                                  (' FaSt ', 'priority')]:
            with self.subTest(setting=setting):
                requests = []
                def handler(request):
                    requests.append(json.loads(request.content))
                    return successful_response()
                client = self.make_client(handler, service_tier=setting)
                client.get_response('hello')
                if expected is None:
                    self.assertNotIn('service_tier', requests[0])
                else:
                    self.assertEqual(requests[0]['service_tier'], expected)

    def test_logs_report_actual_downgrade_or_missing_tier(self):
        for actual, expected in [('default', 'default'), (None, 'not reported')]:
            with self.subTest(actual=actual):
                client = self.make_client(lambda request: successful_response(actual),
                                          service_tier='fast')
                with self.assertLogs('HAL', level='INFO') as logs:
                    client.get_response('hello')
                self.assertIn(f'requested=fast; used={expected}', logs.output[0])

    def test_invalid_tier_fails_before_creating_api_client(self):
        with patch('llm_client.OpenAI') as create_client:
            with self.assertRaisesRegex(ValueError, 'LLM_SERVICE_TIER'):
                LLMClient('openai', 'gpt-6-luna', service_tier='fats',
                          openai_api_key='test-placeholder-not-a-real-key')
            create_client.assert_not_called()

    def test_rejected_tier_is_actionable_without_retry_or_history_change(self):
        for status in (400, 403):
            with self.subTest(status=status):
                requests = []
                def handler(request):
                    requests.append(json.loads(request.content))
                    return error_response(status, 'invalid_value', 'invalid_request_error',
                                          param='service_tier')
                client = self.make_client(handler, service_tier='fast')
                with self.assertRaisesRegex(LLMServiceError, 'LLM_SERVICE_TIER=default') as caught:
                    client.get_response('hello')
                self.assertNotIn('raw server message', str(caught.exception))
                self.assertEqual(len(requests), 1)
                self.assertEqual(client.chat_history, [])

    def test_credit_failure_is_not_retried_and_history_recovers(self):
        requests = []
        def handler(request):
            requests.append(json.loads(request.content))
            if len(requests) == 1:
                return error_response(429, 'credit_balance_exhausted', 'insufficient_quota')
            return successful_response()
        client = self.make_client(handler)
        previous = [{'role': role, 'content': f'{role} turn {turn}'}
                    for turn in range(2) for role in ('user', 'assistant')]
        client.chat_history = copy.deepcopy(previous)
        with self.assertRaisesRegex(LLMServiceError, 'credits are exhausted') as caught:
            client.get_response('failed request')
        self.assertIn('platform.openai.com/settings/organization/billing/', str(caught.exception))
        self.assertEqual(len(requests), 1)
        self.assertEqual(client.chat_history, previous)
        self.assertEqual(client.get_response('new request'), 'I am ready.')
        self.assertEqual(len(requests), 2)
        self.assertNotIn('failed request', json.dumps(requests[1]['messages']))
        self.assertIn('new request', client.chat_history[-2]['content'])
        self.assertEqual(client.chat_history[-1], {'role': 'assistant', 'content': 'I am ready.'})

    def test_service_errors_have_actionable_distinct_messages(self):
        cases = [
            (429, 'rate_limit_exceeded', 'rate_limit_error', 'temporarily rate limiting'),
            (429, 'insufficient_quota', 'insufficient_quota', 'Check credits and limits'),
            (429, 'project_spend_limit_exceeded', 'insufficient_quota', 'account limit'),
            (401, 'invalid_api_key', 'invalid_request_error', 'Check OPENAI_API_KEY'),
            (404, 'model_not_found', 'invalid_request_error', 'Check LLM_MODEL'),
            (503, 'server_is_overloaded', 'server_error', 'temporarily unavailable'),
            (400, 'invalid_parameter', 'invalid_request_error', 'HTTP 400'),
        ]
        for status, code, error_type, expected in cases:
            with self.subTest(code=code):
                client = self.make_client(lambda request:
                    error_response(status, code, error_type))
                with self.assertRaisesRegex(LLMServiceError, expected) as caught:
                    client.get_response('question')
                self.assertNotIn('raw server message', str(caught.exception))
                self.assertEqual(client.chat_history, [])

    def test_connection_and_timeout_errors_are_recoverable(self):
        for error_type, expected in [(httpx.ConnectError, 'network connection'),
                                     (httpx.ReadTimeout, 'timed out')]:
            with self.subTest(error_type=error_type):
                def handler(request):
                    raise error_type('fixture network failure', request=request)
                client = self.make_client(handler)
                with self.assertRaisesRegex(LLMServiceError, expected):
                    client.get_response('question')
                self.assertEqual(client.chat_history, [])

    def test_programming_errors_are_not_hidden(self):
        client = self.make_client(lambda request: successful_response())
        with patch.object(client.client.chat.completions, 'create', side_effect=ValueError('bug')):
            with self.assertRaisesRegex(ValueError, 'bug'):
                client.get_response('question')
        self.assertEqual(client.chat_history, [])

    def test_ollama_success_preserves_existing_behavior(self):
        client = LLMClient('ollama', 'test-model')
        with patch('llm_client.subprocess.run', return_value=types.SimpleNamespace(stdout=' Ready. ')):
            self.assertEqual(client.get_response('hello'), 'Ready.')
        self.assertEqual([turn['role'] for turn in client.chat_history], ['user', 'assistant'])


class SimulatedClock:
    def __init__(self):
        self.now = 0.

    def perf_counter(self):
        return self.now

    def advance(self, seconds, result=None):
        self.now += seconds
        return result


class MainLoopTests(unittest.TestCase):
    def fixture(self):
        namespace = {name: Mock() for name in (
            'logger', 'led', 'voice_input', 'stt', 'llm', 'DisplayServerManager',
            'play_audio', 'handle_api_call', 'time')}
        namespace.update({
            'os': types.SimpleNamespace(getenv=lambda name, default=None: default),
            'sys': sys, 're': re, 'shlex': shlex, 'DEBUG_ON': False, 'DEBUG_PLAYBACK': False,
            'normalize_audio': lambda audio: audio,
            'CommandTooLongError': type('CommandTooLongError', (RuntimeError,), {}),
            'LLMServiceError': LLMServiceError,
            'TranscriptionError': TranscriptionError,
            'AudioOverflowError': AudioOverflowError,
            'FollowupSession': FollowupSession, 'followup_settings': FollowupSettings(),
            'explicitly_addresses_hal': explicitly_addresses_hal,
            'time': types.SimpleNamespace(perf_counter=lambda: 0., sleep=Mock()),
        })
        namespace['stt'].transcribe.return_value = 'Hey HAL, are you there?'
        namespace['stt'].mode = 'static'
        namespace['voice_input'].last_speech_end_at = None
        namespace['handle_api_call'].return_value = 'service result'
        return namespace, load_hal_function('run', namespace)

    def test_service_failure_reopens_listening_without_stopping_display(self):
        for followup in (False, True):
            with self.subTest(followup=followup):
                ns, run = self.fixture()
                display_mgr = ns['DisplayServerManager'].return_value
                failure = LLMServiceError('OpenAI API credits are exhausted.')
                ns['llm'].get_response.side_effect = (
                    ['[EXTERNAL_API_CALL] weather default', failure] if followup else [failure])
                reads = []
                def read_command(on_trigger):
                    reads.append(True)
                    if len(reads) == 1:
                        on_trigger('spacebar')
                        return [0.1], 16000
                    display_mgr.stop.assert_not_called()
                    ns['led'].off.assert_called_once()
                    raise KeyboardInterrupt
                ns['voice_input'].read_command.side_effect = read_command
                with self.assertRaises(SystemExit) as stopped:
                    run()
                self.assertEqual(stopped.exception.code, 0)
                self.assertEqual(len(reads), 2)
                display_mgr.stop.assert_called_once()  # only on deliberate Ctrl-C
                ns['logger'].display.assert_any_call(f'HAL: {failure}')

    def test_unexpected_failure_still_cleans_up_and_surfaces(self):
        ns, run = self.fixture()
        ns['voice_input'].read_command.return_value = ([0.1], 16000)
        ns['llm'].get_response.side_effect = ValueError('unexpected bug')
        with self.assertRaisesRegex(ValueError, 'unexpected bug'):
            run()
        ns['DisplayServerManager'].return_value.stop.assert_called_once()
        ns['led'].off.assert_called_once()

    def test_live_turn_is_wired_to_transcription_and_closed_on_success_failure_and_interrupt(self):
        for scenario in ('success', 'transcription_failure', 'overflow'):
            with self.subTest(scenario=scenario):
                ns, run = self.fixture()
                ns.update(wave=MagicMock(), voice=Mock(), syn_config=object(), USER='fixture',
                          strip_name_at_sentence_end=lambda text, name: text, sf=Mock())
                ns['sf'].read.return_value = ([.1], 16000)
                ns['stt'].mode = 'live'
                streams = [Mock(), Mock()]
                ns['stt'].create_live_stream.side_effect = streams
                ns['llm'].get_response.return_value = 'I am ready.'
                if scenario == 'transcription_failure':
                    ns['stt'].transcribe.side_effect = TranscriptionError('Live service unavailable.')
                reads = []
                def read_command(on_trigger, audio_stream):
                    reads.append(audio_stream)
                    self.assertIs(audio_stream, streams[len(reads) - 1])
                    ns['DisplayServerManager'].return_value.stop.assert_not_called()
                    if len(reads) == 2:
                        streams[0].close.assert_called_once()
                        raise KeyboardInterrupt
                    on_trigger('wakeword')
                    if scenario == 'overflow':
                        raise AudioOverflowError('input overflow')
                    ns['voice_input'].last_speech_end_at = -.8
                    return [.1], 16000
                ns['voice_input'].read_command.side_effect = read_command
                with self.assertRaises(SystemExit) as stopped:
                    run()
                self.assertEqual(stopped.exception.code, 0)
                for stream in streams:
                    stream.close.assert_called_once()
                if scenario == 'overflow':
                    ns['stt'].transcribe.assert_not_called()
                else:
                    ns['stt'].transcribe.assert_called_once_with([.1], 16000, live_stream=streams[0])
                if scenario == 'success':
                    ns['llm'].get_response.assert_called_once_with('Hey HAL, are you there?')
                    self.assertEqual(ns['play_audio'].call_args.kwargs['speech_ended_at'], -.8)
                else:
                    ns['llm'].get_response.assert_not_called()
                    ns['play_audio'].assert_not_called()
                ns['DisplayServerManager'].return_value.stop.assert_called_once()

    def test_repeated_input_overflows_keep_display_up_and_never_transcribe(self):
        ns, run = self.fixture()
        display_mgr = ns['DisplayServerManager'].return_value
        reads = []
        def read_command(on_trigger):
            reads.append(True)
            display_mgr.stop.assert_not_called()
            if len(reads) <= 2:
                on_trigger('wakeword')
                raise AudioOverflowError('input overflow')
            self.assertEqual(ns['led'].off.call_count, 2)
            self.assertEqual(ns['time'].sleep.call_count, 2)
            ns['stt'].transcribe.assert_not_called()
            ns['llm'].get_response.assert_not_called()
            raise KeyboardInterrupt
        ns['voice_input'].read_command.side_effect = read_command
        with self.assertRaises(SystemExit) as stopped:
            run()
        self.assertEqual(stopped.exception.code, 0)
        self.assertEqual(len(reads), 3)
        display_mgr.stop.assert_called_once()

    def test_stage_timings_exclude_idle_and_reset_between_requests_with_debug_off(self):
        ns, run = self.fixture()
        clock = SimulatedClock()
        ns.update(time=clock, wave=MagicMock(), voice=Mock(), syn_config=object(),
                  USER='fixture', strip_name_at_sentence_end=lambda text, name: text,
                  sf=Mock())
        ns['sf'].read.return_value = ([.1], 16000)
        ns['stt'].transcribe.side_effect = lambda *args: clock.advance(1., 'Hey Hal, are you there?')
        replies = iter(['[EXTERNAL_API_CALL] calendar_next_event', 'I am ready.', 'I am ready.'])
        ns['llm'].get_response.side_effect = lambda text: clock.advance(2., next(replies))
        ns['handle_api_call'].side_effect = lambda *args: clock.advance(.25, 'calendar result')
        ns['voice'].synthesize_wav.side_effect = lambda *a, **kw: clock.advance(.3)
        captures = []
        def read_command(on_trigger):
            if len(captures) == 2:
                raise KeyboardInterrupt
            clock.advance(100.)  # Waiting for the user must not count as response time.
            on_trigger('wakeword')
            triggered = clock.now
            clock.advance(.5)
            captures.append((triggered, clock.now))
            return [.1], 16000
        ns['voice_input'].read_command.side_effect = read_command
        playback_elapsed = []
        def play(filename, *, label, triggered_at, capture_ready_at, speech_ended_at):
            self.assertIsNone(speech_ended_at)
            self.assertEqual((triggered_at, capture_ready_at), captures[-1])
            playback_elapsed.append((label, clock.now - triggered_at))
            clock.advance(.6)
        ns['play_audio'].side_effect = play
        with self.assertRaises(SystemExit):
            run()
        self.assertEqual([label for label, _ in playback_elapsed], ['acknowledgment', 'reply', 'reply'])
        for (_, actual), expected in zip(playback_elapsed, [3.5, 6.65, 3.8]):
            self.assertAlmostEqual(actual, expected)
        messages = [call.args[0] % call.args[1:] for call in ns['logger'].info.call_args_list]
        for expected in ('Timing: query transcription 1.000s.',
                         'Timing: initial LLM response 2.000s.',
                         'Timing: external request calendar_next_event 0.250s.',
                         'Timing: follow-up LLM response 2.000s.',
                         'Timing: voice synthesis 0.300s.'):
            self.assertIn(expected, messages)
        self.assertEqual(messages.count('Timing: query transcription 1.000s.'), 2)
        self.assertFalse(ns['DEBUG_ON'])
        self.assertFalse(ns['DEBUG_PLAYBACK'])
        self.assertNotIn('last_command.wav', [call.args[0] for call in ns['sf'].write.call_args_list])


class PlaybackTimingTests(unittest.TestCase):
    def test_start_is_logged_after_stream_open_and_before_waiting_for_audio(self):
        clock = SimulatedClock()
        clock.now = 10.
        audio = Mock()
        audio.high_pass_filter.return_value = audio
        audio.export.side_effect = lambda *a, **kw: clock.advance(.2)
        sd, logger = Mock(), Mock()
        ns = {'time': clock, 'logger': logger, 'sd': sd, 'np': np, 'io': __import__('io'),
              'HI_PASS_FREQ': 0, 'AudioSegment': Mock(), 'sf': Mock(),
              'get_default_device': lambda kind: (None, 16000)}
        ns['AudioSegment'].from_file.return_value = audio
        ns['sf'].read.return_value = (np.ones((16000, 1), dtype=np.float32), 16000)
        def start(*args, **kwargs):
            logger.info.assert_not_called()
            clock.advance(.1)
        sd.play.side_effect = start
        def wait():
            messages = [call.args[0] % call.args[1:] for call in logger.info.call_args_list]
            self.assertIn('Timing: reply playback started; preparation 0.200s, stream startup 0.100s, audio duration 1.000s.', messages)
            self.assertIn('Timing: trigger to reply playback start 5.300s.', messages)
            self.assertIn('Timing: capture ready to reply playback start 4.300s.', messages)
            self.assertIn('Timing: estimated speech end to reply playback start 4.800s.', messages)
            clock.advance(1.)
        sd.wait.side_effect = wait
        play = load_hal_function('play_audio', ns)
        play('fixture.wav', label='reply', triggered_at=5., capture_ready_at=6., speech_ended_at=5.5)
        self.assertEqual(logger.info.call_args.args[0] % logger.info.call_args.args[1:],
                         'Timing: reply playback finished; stream wait 1.000s.')


if __name__ == '__main__':
    unittest.main()
