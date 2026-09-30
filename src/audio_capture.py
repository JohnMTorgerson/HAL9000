"""Continuous input and bounded history, separate from model inference."""
import math
import threading
import time
import numpy as np
from scipy.signal import resample_poly

RATE = 16000

class AudioOverflowError(RuntimeError):
    """Input samples were lost; discard this capture and reopen the microphone."""


class AudioHistory:
    def __init__(self, rate, seconds=60):
        self.rate = int(rate)
        self.data = np.empty(int(rate * seconds), dtype=np.int16)
        self.total = 0
        self.last_wall = None
        self.lock = threading.Lock()

    def append(self, samples, wall=None):
        samples = np.asarray(samples, dtype=np.int16).reshape(-1)
        with self.lock:
            self.total += len(samples)
            count = min(len(samples), len(self.data))
            samples = samples[-count:] if count else samples
            start = (self.total - count) % len(self.data)
            first = min(count, len(self.data) - start)
            self.data[start:start + first] = samples[:first]
            self.data[:count - first] = samples[first:]
            self.last_wall = time.perf_counter() if wall is None else wall

    def position(self):
        with self.lock:
            return self.total, self.last_wall

    def read(self, start, end=None):
        with self.lock:
            end = self.total if end is None else end
            if start < max(0, self.total - len(self.data)):
                raise RuntimeError('Required microphone audio left the history buffer; refusing to truncate the command silently.')
            if start < 0 or end < start or end > self.total:
                raise ValueError('Invalid audio interval')
            count = end - start
            index = start % len(self.data)
            first = min(count, len(self.data) - index)
            return np.concatenate((self.data[index:index + first], self.data[:count - first])).copy()

    def sample_at(self, wall):
        with self.lock:
            if self.last_wall is None:
                return 0
            return max(0, min(self.total, self.total + round((wall - self.last_wall) * self.rate)))

    def time_at(self, sample):
        """Estimate sample time from the latest read; device buffering is separate."""
        with self.lock:
            if self.last_wall is None or not 0 <= sample <= self.total:
                return None
            return self.last_wall - (self.total - sample) / self.rate


def to_audio(samples, rate):
    audio = np.asarray(samples, dtype=np.float32) / 32768
    if rate != RATE:
        divisor = math.gcd(int(rate), RATE)
        audio = resample_poly(audio, RATE // divisor, int(rate) // divisor)
    # Match the tested wake input's 16-bit resampling quantization.
    pcm = np.rint(np.clip(audio, -1, 32767 / 32768) * 32768).astype(np.int16)
    return pcm.astype(np.float32) / 32768


class MicrophoneCapture:
    def __init__(self, device=None, channel=1, stream_factory=None, device_info=None,
                 latency=.25):
        if stream_factory is None or device_info is None:
            import sounddevice as sd
            stream_factory = stream_factory or sd.InputStream
            device_info = device_info or sd.query_devices(device, 'input')
        if not 1 <= channel <= device_info['max_input_channels']:
            raise ValueError('HAL_INPUT_CHANNEL is not available on the microphone.')
        self.rate = int(device_info['default_samplerate'])
        self.device_name = device_info['name']
        self.channel = channel
        self.history = AudioHistory(self.rate)
        self.stop_event = threading.Event()
        self.thread = None
        self.error = None
        self.opened_at = None
        self.stream = stream_factory(device=device, samplerate=self.rate, channels=channel,
                                     dtype='int16', latency=latency)
        self.latency = getattr(self.stream, 'latency', latency)

    def start(self):
        self.stream.start()
        self.opened_at = time.perf_counter()
        self.thread = threading.Thread(target=self._read, name='HAL microphone', daemon=True)
        self.thread.start()

    def _read(self):
        try:
            while not self.stop_event.is_set():
                data, overflow = self.stream.read(max(1, round(.02 * self.rate)))
                if overflow:
                    raise AudioOverflowError('Microphone input overflow: audio was lost; command discarded.')
                self.history.append(data[:, self.channel - 1].copy())
        except Exception as exc:
            if not self.stop_event.is_set():
                self.error = exc

    def check(self):
        if self.error:
            raise self.error
        if self.thread is not None and not self.thread.is_alive():
            raise RuntimeError('Microphone reader stopped unexpectedly.')
        _, last = self.history.position()
        if time.perf_counter() - (last or self.opened_at or time.perf_counter()) > 5:
            raise RuntimeError('No microphone samples received for five seconds.')

    def close(self):
        self.stop_event.set()
        # Keep the stream running until the outstanding 20ms read completes.
        # Aborting first can strand a blocking read on CoreAudio (Mac live test).
        if self.thread is not None:
            self.thread.join(timeout=1)
        try:
            if self.thread is None or not self.thread.is_alive():
                self.stream.stop()
            else:
                self.stream.abort()
        finally:
            self.stream.close()
            if self.thread is not None:
                self.thread.join(timeout=2)
        if self.thread is not None and self.thread.is_alive():
            raise RuntimeError('Microphone reader did not stop after stream closure.')
