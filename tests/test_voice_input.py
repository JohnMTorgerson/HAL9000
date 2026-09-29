"""Microphone/control-flow tests without microphone, models, or account access."""
import logging
from pathlib import Path
import sys
import threading
import time
import types
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from audio_capture import AudioHistory, MicrophoneCapture, to_audio
from spacebar_trigger import SpacebarTrigger
from voice_input import VoiceInput, VoiceSettings, CommandTooLongError
from wake_detector import matches_wake

LOG = logging.getLogger('voice-test')

class CaptureTests(unittest.TestCase):
    def test_history_wrap_does_not_silently_lose_command_start(self):
        h = AudioHistory(10, 2)
        h.append(np.arange(35), wall=3.5)
        np.testing.assert_array_equal(h.read(15, 35), np.arange(15, 35))
        self.assertEqual(h.sample_at(3.0), 30)
        with self.assertRaises(RuntimeError):
            h.read(14)

    def test_shutdown_waits_for_reader_before_stopping_stream(self):
        calls = []
        class Stream:
            def __init__(self, **kwargs):
                self.entered = threading.Event()
                self.allow_return = threading.Event()
                self.aborted = False
            def start(self):
                pass
            def read(self, frames):
                self.entered.set()
                self.allow_return.wait(2)
                return np.full((frames, 1), 123, dtype=np.int16), False
            def stop(self):
                calls.append('stop')
                self.aborted = True
            def abort(self):
                calls.append('abort')
                self.aborted = True
            def close(self):
                calls.append('close')
        capture = MicrophoneCapture(stream_factory=Stream,
            device_info={'name': 'fixture', 'default_samplerate': 16000, 'max_input_channels': 1})
        capture.start()
        self.assertTrue(capture.stream.entered.wait(1))
        finished = threading.Event()
        failure = []
        def close():
            try:
                capture.close()
            except Exception as exc:
                failure.append(exc)
            finally:
                finished.set()
        closer = threading.Thread(target=close)
        closer.start()
        time.sleep(.05)
        # Regression for CoreAudio: the stream is still active while its
        # outstanding read returns, rather than aborted from under that read.
        self.assertEqual(calls, [])
        capture.stream.allow_return.set()
        self.assertTrue(finished.wait(2))
        closer.join()
        self.assertEqual(failure, [])
        self.assertEqual(calls, ['stop', 'close'])
        self.assertGreater(capture.history.position()[0], 0)
        self.assertFalse(capture.thread.is_alive())

    def test_reader_reports_overflow(self):
        class Stream:
            def __init__(self, **kwargs):
                pass
            def start(self):
                pass
            def read(self, frames):
                return np.zeros((frames, 1), dtype=np.int16), True
            def stop(self):
                pass
            def close(self):
                pass
        capture = MicrophoneCapture(stream_factory=Stream,
            device_info={'name': 'fixture', 'default_samplerate': 16000, 'max_input_channels': 1})
        try:
            capture.start()
            capture.thread.join(timeout=1)
            with self.assertRaisesRegex(RuntimeError, 'overflow'):
                capture.check()
        finally:
            capture.close()

    def test_resample_preserves_duration_and_finite_audio(self):
        x = np.full(48000, 1234, dtype=np.int16)
        result = to_audio(x, 48000)
        self.assertEqual(len(result), 16000)
        self.assertTrue(np.isfinite(result).all())
        self.assertAlmostEqual(float(np.median(result)), 1234 / 32768, places=4)

    def test_exact_word_boundaries_and_variants(self):
        for word in ('Hal', 'Hall', 'Hell', 'How', 'Al', 'Howl'):
            self.assertTrue(matches_wake(f'HEY, {word}! What time is it?'))
        for text in ('Hey Alfred', 'Hey Halley', 'Hey Howard', 'Hey howling', 'Hey, what is it?', 'Okay, I will'):
            self.assertFalse(matches_wake(text))

    def test_press_and_release_remain_available_after_decoding(self):
        keys = SpacebarTrigger(LOG)
        keys.press()
        keys.release()
        pressed, released = keys.snapshot()
        self.assertIsNotNone(pressed)
        self.assertGreaterEqual(released, pressed)
        keys.press()  # a held-key repeat must not erase release or reset start
        self.assertEqual(keys.snapshot(), (pressed, released))

class Clock:
    def __init__(self):
        self.now = 0.
        self.capture = None
    def perf_counter(self):
        return self.now
    def sleep(self, seconds):
        self.now += seconds
        if self.capture:
            self.capture.feed()

class FlowTests(unittest.TestCase):
    def fixture(self, kind='wake', endless=False):
        clock = Clock()
        signal = np.zeros(60 * 16000, dtype=np.int16)
        signal[3200:(50 if endless else 1) * 16000] = 1234
        state = {'closed': False, 'keys_closed': False, 'windows': []}
        class Capture:
            def __init__(self, **kwargs):
                self.history = AudioHistory(16000)
                self.rate = 16000
                self.device_name = 'fixture'
                clock.capture = self
            def feed(self):
                total = self.history.position()[0]
                end = min(len(signal), round(clock.now * 16000))
                if end > total:
                    self.history.append(signal[total:end], wall=clock.now)
            def start(self):
                pass
            def check(self):
                self.feed()
            def close(self):
                state['closed'] = True
        class Keys:
            available = True
            def __init__(self, logger):
                pass
            def start(self):
                pass
            def snapshot(self):
                if kind == 'spacebar' and clock.now >= .95:
                    return .9, .95  # both occur during a deliberately slow scan
                return None, None
            def close(self):
                state['keys_closed'] = True
        class Detector:
            def analyze(self, audio):
                state['windows'].append(audio.copy())
                before = clock.capture.history.position()[0]
                clock.sleep(.4)
                state['capture_during_decode'] = clock.capture.history.position()[0] > before
                return {'matched': True, 'models': ['base.en'], 'seconds': .4,
                        'transcripts': {'base.en': 'Hey Hal, how many'}}
            def last_speech_sample(self, audio):
                values = np.flatnonzero(audio)
                return int(values[-1] + 1) if len(values) else None
        voice = VoiceInput(VoiceSettings(maximum=8), LOG, Detector(),
                           capture_factory=Capture, keyboard_factory=Keys)
        return clock, signal, state, voice

    def test_wake_retains_audio_while_detection_runs(self):
        clock, signal, state, voice = self.fixture()
        triggered = []
        with patch('voice_input.time', clock):
            audio, rate = voice.read_command(triggered.append)
        self.assertEqual(triggered, ['wakeword'])
        self.assertEqual(rate, 16000)
        np.testing.assert_array_equal(audio[:16000], signal[:16000].astype(np.float32) / 32768)
        self.assertTrue(state['capture_during_decode'])
        self.assertTrue(state['closed'] and state['keys_closed'])

    def test_short_spacebar_press_during_decode_has_priority_and_finishes(self):
        clock, signal, state, voice = self.fixture('spacebar')
        triggered = []
        with patch('voice_input.time', clock):
            audio, rate = voice.read_command(triggered.append)
        self.assertEqual(triggered, ['spacebar'])
        self.assertLess(clock.now, 2)
        self.assertGreater(len(audio), 0)
        self.assertTrue(state['closed'] and state['keys_closed'])

    def test_overlong_command_is_discarded_and_resources_close(self):
        clock, signal, state, voice = self.fixture(endless=True)
        with patch('voice_input.time', clock):
            with self.assertRaises(CommandTooLongError):
                voice.read_command()
        self.assertTrue(state['closed'] and state['keys_closed'])

    def test_missing_model_still_constructs_manual_input(self):
        fake = types.SimpleNamespace(WhisperWakeDetector=lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('missing')))
        with patch.dict(sys.modules, {'wake_detector': fake}), patch.dict('os.environ', {'WAKE_ENABLED': 'true'}):
            with self.assertLogs('voice-test', level='ERROR'):
                voice = VoiceInput.from_env(LOG)
        self.assertIsNone(voice.detector)

if __name__ == '__main__':
    unittest.main()
