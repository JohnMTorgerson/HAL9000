"""Microphone/control-flow tests without microphone, models, or account access."""
import logging
from pathlib import Path
import sys
import threading
import time
import types
import unittest
from unittest.mock import Mock, patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from audio_capture import AudioHistory, AudioOverflowError, MicrophoneCapture, to_audio
from spacebar_trigger import SpacebarTrigger
from voice_input import VoiceInput, VoiceSettings, CommandTooLongError
from wake_detector import WhisperWakeDetector, matches_wake

LOG = logging.getLogger('voice-test')

class WakeConfigurationTests(unittest.TestCase):
    def test_beam_setting_reaches_warmup_and_each_live_model(self):
        for beam in (1, 2, 5):
            with self.subTest(beam=beam):
                models = []
                def make_model(*args, **kwargs):
                    model = Mock()
                    model.transcribe.side_effect = lambda *a, **kw: (
                        iter([types.SimpleNamespace(text='Hey Hal, what time is it?')]), None)
                    models.append(model)
                    return model
                fake_whisper = types.SimpleNamespace(WhisperModel=make_model)
                fake_vad = types.SimpleNamespace(VadOptions=Mock(),
                    get_speech_timestamps=lambda *a: [{'start': 0, 'end': 16000}])
                with patch.dict(sys.modules, {'faster_whisper': fake_whisper, 'faster_whisper.vad': fake_vad}), \
                     patch('wake_detector.model_path', return_value=Path('/fixture')):
                    detector = WhisperWakeDetector(('tiny.en', 'base.en'), beam_size=beam)
                    result = detector.analyze(np.ones(16000, dtype=np.float32) * .1)
                self.assertTrue(result['matched'])
                self.assertEqual(result['models'], ['tiny.en', 'base.en'])
                for model in models:
                    self.assertEqual(model.transcribe.call_count, 2)
                    self.assertEqual([call.kwargs['beam_size'] for call in model.transcribe.call_args_list],
                                     [beam, beam])

    def test_default_and_env_override_reach_detector(self):
        factory = Mock()
        fake = types.SimpleNamespace(WhisperWakeDetector=factory)
        for env, expected in [({}, 2), ({'WAKE_BEAM_SIZE': '1'}, 1), ({'WAKE_BEAM_SIZE': '5'}, 5)]:
            with self.subTest(expected=expected), patch.dict('os.environ', env, clear=True), \
                 patch.dict(sys.modules, {'wake_detector': fake}):
                voice = VoiceInput.from_env(LOG)
                self.assertIsNotNone(voice.detector)
                self.assertEqual(factory.call_args.kwargs['beam_size'], expected)
        for value in ('0', '11', '1.5', 'nan'):
            with self.subTest(value=value), patch.dict('os.environ', {'WAKE_BEAM_SIZE': value}):
                with self.assertRaises(ValueError):
                    VoiceSettings.from_env()


class CaptureTests(unittest.TestCase):
    def test_history_wrap_does_not_silently_lose_command_start(self):
        h = AudioHistory(10, 2)
        h.append(np.arange(35), wall=3.5)
        np.testing.assert_array_equal(h.read(15, 35), np.arange(15, 35))
        self.assertEqual(h.sample_at(3.0), 30)
        self.assertEqual(h.time_at(30), 3.0)
        self.assertIsNone(h.time_at(36))
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
        options = {}
        calls = []
        class Stream:
            def __init__(self, **kwargs):
                options.update(kwargs)
                self.latency = .3
            def start(self):
                pass
            def read(self, frames):
                return np.zeros((frames, 1), dtype=np.int16), True
            def stop(self):
                calls.append('stop')
            def close(self):
                calls.append('close')
        capture = MicrophoneCapture(stream_factory=Stream,
            device_info={'name': 'fixture', 'default_samplerate': 16000, 'max_input_channels': 1})
        try:
            capture.start()
            capture.thread.join(timeout=1)
            with self.assertRaisesRegex(AudioOverflowError, 'overflow'):
                capture.check()
            self.assertEqual(capture.history.position()[0], 0)
            self.assertEqual(options['latency'], .25)
            self.assertEqual(capture.latency, .3)
        finally:
            capture.close()
        self.assertEqual(calls, ['stop', 'close'])
        self.assertFalse(capture.thread.is_alive())

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
                state['capture_options'] = kwargs
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

    def test_live_audio_starts_after_trigger_and_matches_complete_capture(self):
        for kind in ('wake', 'spacebar'):
            with self.subTest(kind=kind):
                clock, signal, state, voice = self.fixture(kind)
                stream = Mock()
                chunks, triggered = [], []
                def start(rate):
                    self.assertTrue(triggered)
                    self.assertEqual(rate, 16000)
                    self.assertFalse(state['closed'])
                def append(chunk):
                    self.assertFalse(state['closed'])
                    chunks.append(chunk.copy())
                stream.start.side_effect = start
                stream.append.side_effect = append
                with patch('voice_input.time', clock):
                    audio, rate = voice.read_command(triggered.append, audio_stream=stream)
                np.testing.assert_array_equal(np.concatenate(chunks).astype(np.float32) / 32768, audio)
                stream.start.assert_called_once()
                stream.end_audio.assert_called_once()
                stream.cancel.assert_not_called()
                self.assertAlmostEqual(voice.last_speech_end_at, .75)  # speech ends at 1s, minus .25s input latency

    def test_live_capture_failure_cancels_without_committing_partial_audio(self):
        clock, signal, state, voice = self.fixture(endless=True)
        stream = Mock()
        voice.last_speech_end_at = -123
        with patch('voice_input.time', clock):
            with self.assertRaises(CommandTooLongError):
                voice.read_command(audio_stream=stream)
        self.assertTrue(stream.append.called)
        stream.end_audio.assert_not_called()
        stream.cancel.assert_called_once()
        self.assertIsNone(voice.last_speech_end_at)
        self.assertTrue(state['closed'] and state['keys_closed'])

    def test_live_overflow_before_trigger_never_starts_a_session(self):
        clock, signal, state, voice = self.fixture()
        stream = Mock()
        voice.detector.analyze = Mock(side_effect=AudioOverflowError('lost audio'))
        with patch('voice_input.time', clock):
            with self.assertRaises(AudioOverflowError):
                voice.read_command(audio_stream=stream)
        stream.start.assert_not_called()
        stream.append.assert_not_called()
        stream.end_audio.assert_not_called()
        stream.cancel.assert_called_once()

    def test_live_overflow_after_trigger_discards_the_unfinished_turn(self):
        clock, signal, state, voice = self.fixture()
        stream = Mock()
        def append(chunk):
            clock.capture.check = Mock(side_effect=AudioOverflowError('lost audio'))
        stream.append.side_effect = append
        with patch('voice_input.time', clock):
            with self.assertRaises(AudioOverflowError):
                voice.read_command(audio_stream=stream)
        stream.start.assert_called_once()
        stream.append.assert_called_once()
        stream.end_audio.assert_not_called()
        stream.cancel.assert_called_once()
        self.assertIsNone(voice.last_speech_end_at)
        self.assertTrue(state['closed'] and state['keys_closed'])

    def test_overflow_during_decode_discards_match_and_next_capture_succeeds(self):
        clock, _, state, voice = self.fixture()
        original_analyze = voice.detector.analyze
        failed_captures = []
        def analyze_with_loss(audio):
            result = original_analyze(audio)
            failed_captures.append(clock.capture)
            clock.capture.check = Mock(side_effect=AudioOverflowError('input overflow'))
            return result
        voice.detector.analyze = analyze_with_loss
        triggered = []
        with patch('voice_input.time', clock):
            with self.assertRaises(AudioOverflowError):
                voice.read_command(triggered.append)
            self.assertEqual(triggered, [])
            self.assertTrue(state['closed'] and state['keys_closed'])
            voice.detector.analyze = original_analyze
            audio, rate = voice.read_command(triggered.append)
        self.assertIsNot(clock.capture, failed_captures[0])
        self.assertEqual(triggered, ['wakeword'])
        self.assertGreater(len(audio), 0)
        self.assertEqual(rate, 16000)

    def test_input_latency_configuration_reaches_stream_and_rejects_invalid_values(self):
        clock, _, state, voice = self.fixture('spacebar')
        with patch.dict('os.environ', {'HAL_INPUT_LATENCY_SECONDS': '0.5'}):
            voice.settings = VoiceSettings.from_env()
        with patch('voice_input.time', clock):
            voice.read_command()
        self.assertEqual(state['capture_options']['latency'], .5)
        for latency in (0, -.1, 2.1, float('nan'), float('inf')):
            with self.subTest(latency=latency), self.assertRaisesRegex(ValueError, 'HAL_INPUT_LATENCY_SECONDS'):
                VoiceSettings(input_latency=latency)

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

    def test_explicit_device_name_or_number_bypasses_automatic_selection(self):
        for setting, expected in [('Virtual Desktop Mic', 'Virtual Desktop Mic'), ('0', 0)]:
            with self.subTest(setting=setting):
                clock, _, state, voice = self.fixture('spacebar')
                voice.device_selector = Mock(side_effect=AssertionError('Override ignored'))
                with patch.dict('os.environ', {'HAL_INPUT_DEVICE': setting}), patch('voice_input.time', clock):
                    voice.read_command()
                self.assertEqual(state['capture_options']['device'], expected)
                voice.device_selector.assert_not_called()

    def test_missing_model_still_constructs_manual_input(self):
        fake = types.SimpleNamespace(WhisperWakeDetector=lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('missing')))
        with patch.dict(sys.modules, {'wake_detector': fake}), patch.dict('os.environ', {'WAKE_ENABLED': 'true'}):
            with self.assertLogs('voice-test', level='ERROR'):
                voice = VoiceInput.from_env(LOG)
        self.assertIsNone(voice.detector)

if __name__ == '__main__':
    unittest.main()
