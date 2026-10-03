"""Streaming audio and service boundaries, with no live account or microphone."""
import base64
import json
from pathlib import Path
import queue
import sys
import threading
import unittest
from unittest.mock import Mock, patch

import httpx
import numpy as np
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from live_transcription import LiveSettings, LiveTranscription, PCM24kEncoder, TranscriptionError
from whisper_stt import WhisperSTT


class ResamplingTests(unittest.TestCase):
    def test_arbitrary_chunks_match_a_single_resample_without_edge_or_duration_drift(self):
        for rate in (16000, 24000, 44100, 48000):
            with self.subTest(rate=rate):
                rng = np.random.default_rng(42)
                samples = rng.integers(-20000, 20000, rate * 2 + 37, dtype=np.int16)
                encoder = PCM24kEncoder(rate)
                data, start = [], 0
                while start < len(samples):
                    size = int(rng.integers(1, 7000))
                    data.append(encoder.encode(samples[start:start + size]))
                    start += size
                data.append(encoder.encode(final=True))
                actual = np.frombuffer(b''.join(data), dtype='<i2')
                expected = resample_poly(samples.astype(np.float32) / 32768, encoder.up, encoder.down)
                expected = np.rint(np.clip(expected, -1, 32767 / 32768) * 32768).astype(np.int16)
                np.testing.assert_array_equal(actual, expected)
                self.assertLessEqual(len(encoder.buffer), encoder.margin * 2 + encoder.down)


class FakeSocket:
    def __init__(self, *, final='Hey Hal, what time is it?', malformed_config=False, automatic_final=True):
        self.messages = []
        self.incoming = queue.Queue()
        self.appended = threading.Event()
        self.committed = threading.Event()
        self.closed = threading.Event()
        self.final = final
        self.malformed_config = malformed_config
        self.automatic_final = automatic_final
        self.send_threads = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def emit(self, event):
        self.incoming.put(json.dumps(event))

    def send(self, raw):
        self.send_threads.append(threading.get_ident())
        event = json.loads(raw)
        self.messages.append(event)
        if event['type'] == 'session.update':
            session = event['session']
            if self.malformed_config:
                session['audio']['input']['turn_detection'] = {'type': 'server_vad'}
            self.emit({'type': 'session.created', 'session': {}})
            self.emit({'type': 'session.updated', 'session': session})
        elif event['type'] == 'input_audio_buffer.append':
            self.appended.set()
            self.emit({'type': 'conversation.item.input_audio_transcription.delta',
                       'item_id': 'our-turn', 'content_index': 0, 'delta': 'An incorrect partial'})
        elif event['type'] == 'input_audio_buffer.commit':
            self.committed.set()
            self.emit({'type': 'input_audio_buffer.committed', 'item_id': 'our-turn'})
            if self.automatic_final:
                self.emit_final()

    def emit_final(self, item_id='our-turn', text=None):
        self.emit({'type': 'conversation.item.input_audio_transcription.completed',
                   'item_id': item_id, 'content_index': 0,
                   'transcript': self.final if text is None else text})

    def recv(self, timeout=None):
        if self.closed.is_set():
            raise OSError('closed fixture')
        try:
            return self.incoming.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError from None

    def close(self):
        self.closed.set()


class StreamingTests(unittest.TestCase):
    def make_stream(self, socket=None, settings=None, connector=None):
        socket = socket or FakeSocket()
        connector = connector or Mock(return_value=socket)
        stream = LiveTranscription('fixture-not-a-key', settings or LiveSettings(timeout=1),
                                   logger=Mock(), connector=connector)
        self.addCleanup(stream.close)
        return stream, socket, connector

    def test_constructor_and_idle_capture_do_not_connect(self):
        stream, socket, connect = self.make_stream()
        connect.assert_not_called()
        stream.close()
        connect.assert_not_called()
        self.assertEqual(socket.messages, [])

    def test_audio_streams_before_finish_and_only_committed_final_is_returned(self):
        stream, socket, connect = self.make_stream()
        samples = np.arange(16000, dtype=np.int16)
        stream.start(16000)
        stream.append(samples[:8000])
        self.assertTrue(socket.appended.wait(2))
        self.assertFalse(socket.committed.is_set())
        stream.append(samples[8000:])
        stream.end_audio()
        self.assertEqual(stream.result(), socket.final)
        stream.close()
        self.assertFalse(stream._thread.is_alive())
        self.assertNotIn(threading.get_ident(), socket.send_threads)
        config = socket.messages[0]['session']
        self.assertEqual(config['type'], 'transcription')
        self.assertEqual(config['audio']['input']['transcription']['model'], 'gpt-live-transcribe')
        self.assertEqual(config['audio']['input']['transcription']['languages'], ['en'])
        self.assertIsNone(config['audio']['input']['turn_detection'])
        self.assertEqual(connect.call_args.args[0], LiveTranscription.URL)
        self.assertEqual(connect.call_args.kwargs['additional_headers'], {'Authorization': 'Bearer fixture-not-a-key'})
        data = b''.join(base64.b64decode(e['audio']) for e in socket.messages if e['type'] == 'input_audio_buffer.append')
        actual = np.frombuffer(data, dtype='<i2')
        reference = resample_poly(samples.astype(np.float32) / 32768, 3, 2)
        reference = np.rint(np.clip(reference, -1, 32767 / 32768) * 32768).astype(np.int16)
        np.testing.assert_array_equal(actual, reference)
        self.assertEqual([e['type'] for e in socket.messages].count('input_audio_buffer.commit'), 1)
        self.assertEqual(socket.messages[-1]['type'], 'input_audio_buffer.commit')

    def test_results_are_matched_by_item_id(self):
        stream, socket, _ = self.make_stream(FakeSocket(automatic_final=False))
        stream.start(24000)
        stream.append(np.ones(2400, dtype=np.int16))
        stream.end_audio()
        self.assertTrue(socket.committed.wait(2))
        socket.emit_final('another-turn', 'wrong result')
        socket.emit_final('our-turn', 'correct result')
        self.assertEqual(stream.result(), 'correct result')

    def test_close_failure_does_not_discard_a_successful_final(self):
        class CloseFailure(FakeSocket):
            def __exit__(self, *args):
                super().__exit__(*args)
                raise OSError('connection lost during close')
        stream, socket, _ = self.make_stream(CloseFailure())
        stream.start(24000)
        stream.append(np.ones(2400, dtype=np.int16))
        stream.end_audio()
        stream._thread.join(timeout=2)
        self.assertFalse(stream._thread.is_alive())
        self.assertEqual(stream.result(), socket.final)

    def test_empty_final_and_server_errors_are_recoverable_without_raw_error_text(self):
        for scenario in ('empty', 'server_error', 'wrong_config'):
            with self.subTest(scenario=scenario):
                socket = FakeSocket(final='', malformed_config=scenario == 'wrong_config',
                                    automatic_final=scenario != 'server_error')
                stream, _, _ = self.make_stream(socket)
                stream.start(16000)
                stream.append(np.ones(1600, dtype=np.int16))
                stream.end_audio()
                if scenario == 'server_error':
                    self.assertTrue(socket.committed.wait(2))
                    socket.emit({'type': 'error', 'error': {'code': 'invalid_value',
                                'message': 'secret fixture credential must never be logged'}})
                with self.assertRaises(TranscriptionError) as caught:
                    stream.result()
                self.assertNotIn('secret fixture', str(caught.exception))
                if scenario == 'wrong_config':
                    self.assertFalse(socket.appended.is_set())

    def test_timeout_rejects_partial_text(self):
        stream, socket, _ = self.make_stream(FakeSocket(automatic_final=False))
        stream.start(16000)
        stream.append(np.ones(1600, dtype=np.int16))
        stream.end_audio()
        with self.assertRaisesRegex(TranscriptionError, 'timed out'):
            stream.result()
        stream.close()
        self.assertFalse(stream._thread.is_alive())

    def test_cancel_during_connection_never_sends_audio_after_connection_completes(self):
        gate, entered = threading.Event(), threading.Event()
        socket = FakeSocket()
        def connect(*args, **kwargs):
            entered.set()
            gate.wait(2)
            return socket
        stream, _, _ = self.make_stream(socket, connector=connect)
        stream.start(16000)
        self.assertTrue(entered.wait(1))
        stream.append(np.ones(1600, dtype=np.int16))
        stream.cancel('discarded command')
        gate.set()
        stream.close()
        self.assertFalse(stream._thread.is_alive())
        self.assertEqual(socket.messages, [])

    def test_network_failure_does_not_block_capture_feed(self):
        connect = Mock(side_effect=OSError('sensitive fixture data'))
        stream, _, _ = self.make_stream(connector=connect)
        stream.start(16000)
        stream.append(np.ones(1600, dtype=np.int16))
        stream.end_audio()
        with self.assertRaises(TranscriptionError) as caught:
            stream.result()
        self.assertNotIn('sensitive fixture', str(caught.exception))
        connect.assert_called_once()

    def test_real_websocket_transport_against_local_fixture(self):
        from websockets.sync.client import connect
        from websockets.sync.server import serve
        frames = []
        def handler(ws):
            for raw in ws:
                event = json.loads(raw)
                frames.append(event)
                if event['type'] == 'session.update':
                    ws.send(json.dumps({'type': 'session.updated', 'session': event['session']}))
                elif event['type'] == 'input_audio_buffer.commit':
                    ws.send(json.dumps({'type': 'input_audio_buffer.committed', 'item_id': 'wire-test'}))
                    ws.send(json.dumps({'type': 'conversation.item.input_audio_transcription.completed',
                                        'item_id': 'wire-test', 'transcript': 'Wire protocol works.'}))
        with serve(handler, '127.0.0.1', 0) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            port = server.socket.getsockname()[1]
            def local_connect(url, **kwargs):
                return connect(f'ws://127.0.0.1:{port}', proxy=None, **kwargs)
            stream, _, _ = self.make_stream(connector=local_connect)
            try:
                stream.start(44100)
                stream.append(np.zeros(4410, dtype=np.int16))
                stream.end_audio()
                self.assertEqual(stream.result(), 'Wire protocol works.')
            finally:
                stream.close()
                server.shutdown()
                thread.join(timeout=2)
        self.assertEqual(frames[-1]['type'], 'input_audio_buffer.commit')


class ModeTests(unittest.TestCase):
    def make_stt(self, env=None, handler=None):
        requests = []
        def response(request):
            requests.append(request)
            return handler(request) if handler else httpx.Response(200, json={'text': 'Static transcript.'})
        client = httpx.Client(transport=httpx.MockTransport(response))
        self.addCleanup(client.close)
        real_openai = __import__('openai').OpenAI
        settings = {'TRANSCRIPTION_BACKEND': 'api', 'OPENAI_API_KEY': 'fixture-not-a-key'}
        settings.update(env or {})
        with patch.dict('os.environ', settings, clear=True), patch('whisper_stt.openai.OpenAI',
            side_effect=lambda **kwargs: real_openai(http_client=client, **kwargs)):
            stt = WhisperSTT(logger=Mock())
        self.addCleanup(stt.client.close)
        return stt, requests

    def test_static_default_never_constructs_live_transport_and_uploads_complete_wav(self):
        with patch('whisper_stt.LiveTranscription', side_effect=AssertionError('unexpected live path')):
            stt, requests = self.make_stt()
            self.assertEqual(stt.mode, 'static')
            self.assertIsNone(stt.create_live_stream())
            self.assertEqual(stt.transcribe(np.ones(16000, dtype=np.float32) * .1), 'Static transcript.')
        self.assertEqual(len(requests), 1)
        self.assertIn(b'gpt-4o-mini-transcribe', requests[0].content)
        self.assertIn(b'RIFF', requests[0].content)
        self.assertIn(b'name="language"\r\n\r\nen\r\n', requests[0].content)

    def test_static_language_can_be_changed_or_omitted(self):
        for value in (' FR ', '', '   '):
            with self.subTest(language=value):
                stt, requests = self.make_stt({'TRANSCRIPTION_LANGUAGE': value})
                stt.transcribe(np.ones(1600, dtype=np.float32) * .1)
                body = requests[0].content
                if value.strip():
                    self.assertIn(b'name="language"\r\n\r\nfr\r\n', body)
                else:
                    self.assertNotIn(b'name="language"', body)


    def test_live_success_does_not_also_upload_a_file(self):
        stt, requests = self.make_stt({'TRANSCRIPTION_MODE': 'live'})
        stream = Mock()
        stream.result.return_value = 'Final live transcript.'
        self.assertEqual(stt.transcribe(np.ones(1600), live_stream=stream), 'Final live transcript.')
        self.assertEqual(requests, [])

    def test_live_failure_only_uploads_static_fallback_when_explicitly_enabled(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                settings = {'TRANSCRIPTION_MODE': 'live'}
                if enabled:
                    settings['LIVE_TRANSCRIPTION_FALLBACK'] = 'true'
                stt, requests = self.make_stt(settings)
                stream = Mock()
                stream.result.side_effect = TranscriptionError('fixture failure')
                if enabled:
                    self.assertEqual(stt.transcribe(np.ones(1600), live_stream=stream), 'Static transcript.')
                    self.assertEqual(len(requests), 1)
                else:
                    with self.assertRaises(TranscriptionError):
                        stt.transcribe(np.ones(1600), live_stream=stream)
                    self.assertEqual(requests, [])
                stream.close.assert_called_once()

    def test_static_service_failure_is_recoverable_and_not_retried(self):
        stt, requests = self.make_stt(handler=lambda request: httpx.Response(429, json={'error': {
            'message': 'private raw data', 'code': 'insufficient_quota', 'type': 'insufficient_quota'}}))
        with self.assertRaisesRegex(TranscriptionError, 'credits') as caught:
            stt.transcribe(np.ones(1600))
        self.assertNotIn('private raw data', str(caught.exception))
        self.assertEqual(len(requests), 1)

    def test_local_mode_is_still_local_and_invalid_modes_do_not_override_it(self):
        whisper = Mock()
        whisper.load_model.return_value.transcribe.return_value = {'text': 'Local transcript.'}
        with patch.dict('os.environ', {'TRANSCRIPTION_BACKEND': 'local'}, clear=True), \
             patch.dict(sys.modules, {'whisper': whisper}), patch('whisper_stt.openai.OpenAI') as api:
            stt = WhisperSTT()
            self.assertEqual(stt.transcribe(np.ones(1600)), 'Local transcript.')
            api.assert_not_called()
            whisper.load_model.assert_called_once_with('base')
        for env, expected in [({'TRANSCRIPTION_MODE': 'invalid'}, 'TRANSCRIPTION_MODE'),
                              ({'TRANSCRIPTION_MODE': 'live', 'TRANSCRIPTION_BACKEND': 'local'}, 'requires'),
                              ({'TRANSCRIPTION_MODE': 'live', 'LIVE_TRANSCRIPTION_DELAY': 'fast'}, 'DELAY')]:
            with self.subTest(env=env), self.assertRaisesRegex(ValueError, expected):
                self.make_stt(env)


if __name__ == '__main__':
    unittest.main()
