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
from unittest.mock import Mock, patch

import httpx
from openai import OpenAI

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))
from audio_devices import choose_input_device
from llm_client import LLMClient, LLMServiceError


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


def error_response(status, code, error_type='server_error'):
    return httpx.Response(status, json={'error': {
        'message': 'This raw server message must not become a HAL reply.',
        'type': error_type, 'param': None, 'code': code}})


def successful_response():
    return httpx.Response(200, json={
        'id': 'chatcmpl-test', 'object': 'chat.completion', 'created': 0,
        'model': 'test-model', 'choices': [{'index': 0, 'finish_reason': 'stop',
            'message': {'role': 'assistant', 'content': 'I am ready.'}}]})


class LLMTests(unittest.TestCase):
    def make_client(self, handler):
        http_client = httpx.Client(transport=httpx.MockTransport(handler))
        self.addCleanup(http_client.close)
        with patch('llm_client.OpenAI', side_effect=lambda **kwargs:
                   OpenAI(http_client=http_client, **kwargs)):
            client = LLMClient('openai', 'test-model', max_history=2,
                               openai_api_key='test-placeholder-not-a-real-key')
        self.addCleanup(client.client.close)
        return client

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


class MainLoopTests(unittest.TestCase):
    def fixture(self):
        namespace = {name: Mock() for name in (
            'logger', 'led', 'voice_input', 'stt', 'llm', 'DisplayServerManager',
            'play_audio', 'handle_api_call')}
        namespace.update({
            'os': types.SimpleNamespace(getenv=lambda name, default=None: default),
            'sys': sys, 're': re, 'shlex': shlex, 'DEBUG_ON': False, 'DEBUG_PLAYBACK': False,
            'normalize_audio': lambda audio: audio,
            'CommandTooLongError': type('CommandTooLongError', (RuntimeError,), {}),
            'LLMServiceError': LLMServiceError,
        })
        namespace['stt'].transcribe.return_value = 'Hey HAL, are you there?'
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


if __name__ == '__main__':
    unittest.main()
