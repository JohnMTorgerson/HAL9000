"""One explicitly triggered audio turn over OpenAI's transcription WebSocket.

The microphone thread never does network IO. Nothing connects or uploads until
start() is called after a wake/spacebar trigger. Partial text is timing data only;
only the final transcript for our committed item can reach the language model.
"""
import base64
from dataclasses import dataclass
import json
import logging
import math
import os
import queue
import threading
import time

import numpy as np
from scipy.signal import resample_poly


class TranscriptionError(RuntimeError):
    """A transcription failure that can return HAL to listening."""


@dataclass(frozen=True)
class LiveSettings:
    model: str = 'gpt-live-transcribe'
    delay: str = 'low'
    languages: tuple = ('en',)
    timeout: float = 8.

    def __post_init__(self):
        if not self.model or self.delay not in ('minimal', 'low', 'medium', 'high', 'xhigh'):
            raise ValueError('Check LIVE_TRANSCRIPTION_MODEL and LIVE_TRANSCRIPTION_DELAY '
                             '(minimal, low, medium, high, or xhigh).')
        if not 1 <= self.timeout <= 60:
            raise ValueError('LIVE_TRANSCRIPTION_TIMEOUT_SECONDS must be between 1 and 60.')

    @classmethod
    def from_env(cls):
        return cls(model=os.getenv('LIVE_TRANSCRIPTION_MODEL', 'gpt-live-transcribe').strip(),
                   delay=os.getenv('LIVE_TRANSCRIPTION_DELAY', 'low').strip().lower(),
                   languages=tuple(s.strip() for s in
                       os.getenv('LIVE_TRANSCRIPTION_LANGUAGES', 'en').split(',') if s.strip()),
                   timeout=float(os.getenv('LIVE_TRANSCRIPTION_TIMEOUT_SECONDS', '8')))


class PCM24kEncoder:
    """Streaming polyphase resampling with preserved filter context and phase.

Keep a small right margin until the next chunk (or flush). Resampling each chunk
independently would add edge transients and rounding drift at 44.1 kHz.
"""
    def __init__(self, input_rate):
        divisor = math.gcd(int(input_rate), 24000)
        self.up, self.down = 24000 // divisor, int(input_rate) // divisor
        self.margin = self.down * math.ceil(12 * max(self.up, self.down) / self.up / self.down)
        self.buffer = np.empty(0, np.int16)
        self.offset = self.total = self.emitted = 0

    def encode(self, samples=(), *, final=False):
        samples = np.asarray(samples, dtype=np.int16).reshape(-1)
        self.buffer = np.concatenate((self.buffer, samples))
        self.total += len(samples)
        end = ((self.total * self.up + self.down - 1) // self.down if final else
               max(0, (self.total - self.margin) // self.down) * self.up)
        if end <= self.emitted:
            return b''
        converted = resample_poly(self.buffer.astype(np.float32) / 32768, self.up, self.down)
        base = self.offset * self.up // self.down
        audio = converted[self.emitted - base:end - base]
        self.emitted = end
        # Offset always remains divisible by down, preserving the global phase.
        keep_from = max(self.offset, ((end * self.down // self.up - self.margin)
                                     // self.down) * self.down)
        self.buffer = self.buffer[keep_from - self.offset:]
        self.offset = keep_from
        return np.rint(np.clip(audio, -1, 32767 / 32768) * 32768).astype('<i2').tobytes()


def _service_failure(error):
    """Never echo raw server messages, which can contain input or credentials."""
    code = error.get('code') if isinstance(error, dict) else getattr(error, 'code', None)
    status = getattr(error, 'status_code', None)
    if status is None:
        status = getattr(getattr(error, 'response', None), 'status_code', None)
    if code in ('credit_balance_exhausted', 'insufficient_quota') or status == 429:
        return TranscriptionError('OpenAI transcription is unavailable. Check API credits and rate limits.')
    if code == 'invalid_api_key' or status == 401:
        return TranscriptionError('OpenAI rejected the transcription API key. Check OPENAI_API_KEY.')
    if code == 'model_not_found' or status in (403, 404):
        return TranscriptionError('OpenAI denied transcription access. Check the model and API project permissions.')
    return TranscriptionError('OpenAI transcription failed. Check the connection, model access, and transcription settings.')


class LiveTranscription:
    URL = 'wss://api.openai.com/v1/realtime?intent=transcription'

    def __init__(self, api_key, settings=None, logger=None, connector=None):
        self.settings = settings or LiveSettings()
        self.logger = logger if logger is not None else logging.getLogger('HAL')
        if connector is None:
            try:
                from websockets.sync.client import connect
            except ImportError:
                raise ValueError('Live transcription needs: python -m pip install -r '
                                 'src/requirements-live.txt (from the repository root).') from None
            connector = connect
        self._connect = connector
        self._key = api_key
        self._queue = queue.Queue(maxsize=512)
        self._stop = threading.Event()
        self._done = threading.Event()
        self._thread = None
        self._socket = None
        self._error = None
        self._text = None
        self._ended = False
        self.started_at = self.audio_ended_at = self.final_at = None
        self._rate = None
        self._input_samples = 0

    def start(self, rate):
        if self._thread is not None:
            raise RuntimeError('A live transcription turn cannot be started twice.')
        self._rate = int(rate)
        if self._rate <= 0:
            raise ValueError('Invalid live audio sample rate.')
        self.started_at = time.perf_counter()
        self._thread = threading.Thread(target=self._run, name='HAL live transcription', daemon=True)
        self._thread.start()

    def append(self, samples):
        if self._stop.is_set() or self._done.is_set():
            return
        if self._thread is None or self._ended:
            raise RuntimeError('Audio must be appended between live start and end_audio.')
        samples = np.asarray(samples, dtype=np.int16).reshape(-1).copy()
        self._input_samples += len(samples)
        if self._input_samples > 30 * self._rate:
            self.cancel('Live audio exceeded the 30-second turn limit.')
            return
        self._put(samples)

    def _put(self, value):
        try:
            self._queue.put_nowait(value)
        except queue.Full:
            self.cancel('Live transcription could not keep up with the audio.')

    def end_audio(self):
        if not self._ended:
            self._ended = True
            self.audio_ended_at = time.perf_counter()
            self._put(None)

    def result(self):
        if self._thread is None or not self._ended:
            raise TranscriptionError('Live transcription did not receive a complete command.')
        remaining = max(0., self.settings.timeout - (time.perf_counter() - self.audio_ended_at))
        if not self._done.wait(remaining):
            self.cancel('Live transcription timed out waiting for the final transcript.')
        if self._error is not None:
            raise self._error
        if not self._text:
            raise TranscriptionError('Live transcription returned no speech. Please repeat the request.')
        return self._text

    def cancel(self, reason='Live transcription was cancelled.'):
        self._stop.set()
        self._error = TranscriptionError(reason)
        self._done.set()

    def close(self):
        self._stop.set()
        if self._socket is not None:
            try:
                self._socket.close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.)

    def _send(self, event):
        if not self._stop.is_set():
            self._socket.send(json.dumps(event))

    def _receive(self, timeout=.02):
        try:
            event = json.loads(self._socket.recv(timeout=timeout))
        except TimeoutError:
            return None
        if not isinstance(event, dict):
            raise TranscriptionError('Invalid live transcription response.')
        if event.get('type') in ('error', 'conversation.item.input_audio_transcription.failed'):
            raise _service_failure(event.get('error', {}))
        return event

    def _run(self):
        try:
            # This logger cannot leak request headers or audio if an application
            # enables global DEBUG logging on the websocket package.
            wire_logger = logging.Logger('HAL live transport', level=logging.CRITICAL + 1)
            with self._connect(self.URL, additional_headers={'Authorization': f'Bearer {self._key}'},
                               open_timeout=5, close_timeout=.5, compression=None,
                               max_size=1024 * 1024, logger=wire_logger) as socket:
                self._socket = socket
                if self._stop.is_set():
                    return
                transcription = {'model': self.settings.model, 'delay': self.settings.delay}
                if self.settings.languages:
                    transcription['languages'] = list(self.settings.languages)
                self._send({'type': 'session.update', 'session': {
                    'type': 'transcription', 'audio': {'input': {
                        'format': {'type': 'audio/pcm', 'rate': 24000},
                        'transcription': transcription, 'turn_detection': None}}}})
                deadline = time.perf_counter() + 5
                while not self._stop.is_set():
                    if time.perf_counter() >= deadline:
                        raise TranscriptionError('Live transcription session setup timed out.')
                    event = self._receive()
                    if event and event.get('type') == 'session.updated':
                        session = event.get('session', {})
                        audio = session.get('audio', {}).get('input', {})
                        audio_format = audio.get('format', {})
                        if (session.get('type') != 'transcription' or
                            audio.get('transcription', {}).get('model') != self.settings.model or
                            audio_format.get('type') != 'audio/pcm' or
                            audio_format.get('rate') != 24000 or
                            audio.get('turn_detection') is not None):
                            raise TranscriptionError('OpenAI did not accept the live transcription configuration.')
                        break
                if self._stop.is_set():
                    return
                self.logger.info('Timing: live transcription session ready %.3fs after trigger.',
                                 time.perf_counter() - self.started_at)
                encoder = PCM24kEncoder(self._rate)
                committed = False
                item_id = None
                finals = {}
                first_delta = False
                while not self._stop.is_set():
                    if self.audio_ended_at is not None and time.perf_counter() - self.audio_ended_at > self.settings.timeout:
                        raise TranscriptionError('Live transcription timed out waiting for the final transcript.')
                    if not committed:
                        try:
                            chunk = self._queue.get_nowait()
                        except queue.Empty:
                            chunk = b''
                        if chunk is None:
                            pcm = encoder.encode(final=True)
                            if pcm:
                                self._send({'type': 'input_audio_buffer.append',
                                            'audio': base64.b64encode(pcm).decode('ascii')})
                            self._send({'type': 'input_audio_buffer.commit'})
                            committed = True
                            self.logger.info('Live transcription: committed %.2fs of audio.',
                                             self._input_samples / self._rate)
                        elif len(chunk):
                            pcm = encoder.encode(chunk)
                            if pcm:
                                self._send({'type': 'input_audio_buffer.append',
                                            'audio': base64.b64encode(pcm).decode('ascii')})
                    event = self._receive()
                    if event:
                        kind = event.get('type')
                        if kind == 'input_audio_buffer.committed':
                            if not committed or not event.get('item_id') or item_id is not None:
                                raise TranscriptionError('Unexpected live transcription turn boundary.')
                            item_id = event['item_id']
                        elif kind == 'conversation.item.input_audio_transcription.delta' and not first_delta:
                            first_delta = True
                            self.logger.info('Timing: live transcription first partial %.3fs after trigger '
                                             '(audio %s).', time.perf_counter() - self.started_at,
                                             'finished' if self._ended else 'still recording')
                        elif kind == 'conversation.item.input_audio_transcription.completed':
                            if not committed or not event.get('item_id'):
                                raise TranscriptionError('Live transcript arrived before the command was committed.')
                            if event.get('content_index', 0) == 0:
                                finals[event['item_id']] = event.get('transcript')
                                if len(finals) > 4:
                                    raise TranscriptionError('Unexpected extra live transcription results.')
                    if item_id in finals:
                        text = finals[item_id]
                        if not isinstance(text, str) or not text.strip():
                            raise TranscriptionError('Live transcription returned no speech. Please repeat the request.')
                        self._text = text.strip()
                        self.final_at = time.perf_counter()
                        self.logger.info('Timing: live transcription final %.3fs after audio end.',
                                         self.final_at - self.audio_ended_at)
                        self._done.set()
                        return
        except Exception as exc:
            # A close-handshake error must not invalidate an already accepted
            # final transcript and accidentally cause another billed upload.
            if not self._stop.is_set() and self._text is None:
                self._error = exc if isinstance(exc, TranscriptionError) else _service_failure(exc)
                self._done.set()
        finally:
            self._socket = None
            self._done.set()
