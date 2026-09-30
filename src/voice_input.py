"""Wake/spacebar capture, returning one complete utterance to HAL's existing STT."""
from dataclasses import dataclass
import os
import time
import numpy as np
from audio_capture import MicrophoneCapture, RATE, to_audio
from spacebar_trigger import SpacebarTrigger
from wake_models import model_names

class CommandTooLongError(RuntimeError):
    """A complete request could not fit inside the existing STT audio limit."""


@dataclass(frozen=True)
class VoiceSettings:
    enabled: bool = True
    models: tuple = ('base.en',)
    window: float = 3.
    hop: float = .75
    silence: float = 1.2
    maximum: float = 25.
    threads: int = 2
    channel: int = 1
    max_gain_db: float = 24.
    normalization: str = 'capped'
    input_latency: float = .25
    beam_size: int = 2

    def __post_init__(self):
        if not (.25 <= self.hop <= self.window <= 6):
            raise ValueError('Use 0.25 <= WAKE_HOP_SECONDS <= WAKE_WINDOW_SECONDS <= 6.')
        if not (.5 <= self.silence <= 3 and 8 <= self.maximum <= 28):
            raise ValueError('Use WAKE_SILENCE_SECONDS 0.5–3 and VOICE_MAX_SECONDS 8–28.')
        if not (1 <= self.threads <= 16 and self.channel >= 1 and 0 <= self.max_gain_db <= 36):
            raise ValueError('Invalid wake thread, microphone channel, or gain setting.')
        if self.normalization not in ('capped', 'peak'):
            raise ValueError('WAKE_NORMALIZATION must be capped or peak.')
        if not (.02 <= self.input_latency <= 2):
            raise ValueError('Use HAL_INPUT_LATENCY_SECONDS 0.02–2.')
        if not (isinstance(self.beam_size, int) and 1 <= self.beam_size <= 10):
            raise ValueError('Use an integer WAKE_BEAM_SIZE from 1 to 10.')

    @classmethod
    def from_env(cls):
        return cls(enabled=os.getenv('WAKE_ENABLED', 'true').lower() not in ('0', 'false', 'no', 'off'),
                   models=model_names(), window=float(os.getenv('WAKE_WINDOW_SECONDS', 3)),
                   hop=float(os.getenv('WAKE_HOP_SECONDS', .75)),
                   silence=float(os.getenv('WAKE_SILENCE_SECONDS', 1.2)),
                   maximum=float(os.getenv('VOICE_MAX_SECONDS', 25)),
                   threads=int(os.getenv('WAKE_THREADS', 2)),
                   channel=int(os.getenv('HAL_INPUT_CHANNEL', 1)),
                   max_gain_db=float(os.getenv('WAKE_MAX_GAIN_DB', 24)),
                   normalization=os.getenv('WAKE_NORMALIZATION', 'capped'),
                   input_latency=float(os.getenv('HAL_INPUT_LATENCY_SECONDS', .25)),
                   beam_size=int(os.getenv('WAKE_BEAM_SIZE', 2)))

class VoiceInput:
    def __init__(self, settings, logger, detector=None, device_selector=lambda: None,
                 capture_factory=MicrophoneCapture, keyboard_factory=SpacebarTrigger):
        self.settings = settings
        self.logger = logger
        self.detector = detector
        self.device_selector = device_selector
        self.capture_factory = capture_factory
        self.keyboard_factory = keyboard_factory
        self.last_speech_end_at = None

    @classmethod
    def from_env(cls, logger, device_selector=lambda: None):
        settings = VoiceSettings.from_env()
        detector = None
        if settings.enabled:
            try:
                from wake_detector import WhisperWakeDetector
                detector = WhisperWakeDetector(settings.models, settings.threads,
                    settings.max_gain_db, settings.normalization, beam_size=settings.beam_size)
                logger.info('Whisper wake detection ready: %s, %.2fs windows / %.2fs interval, '
                            'CPU INT8, beam %s, %s threads.', ', '.join(settings.models),
                            settings.window, settings.hop, settings.beam_size, settings.threads)
            except Exception:
                logger.exception('Wake detector unavailable; spacebar remains usable. From the repository root run '
                                 'python -m pip install -r src/requirements-wake.txt, then python src/setup_wake.py.')
        else:
            logger.info('Wake detection disabled by WAKE_ENABLED; spacebar mode only.')
        return cls(settings, logger, detector, device_selector)

    def read_command(self, on_trigger=lambda kind: None, audio_stream=None):
        settings = self.settings
        self.last_speech_end_at = None
        device = os.getenv('HAL_INPUT_DEVICE')
        if device:
            device = int(device) if device.isdigit() else device
        else:
            device = self.device_selector()
        capture = self.capture_factory(device=device, channel=settings.channel,
                                       latency=settings.input_latency)
        keys = self.keyboard_factory(self.logger)
        stream_cursor = None
        completed = False
        speech_sample = None
        try:
            capture.start()
            keys.start()
            if self.detector is None and not keys.available:
                raise RuntimeError('Neither wake detection nor a spacebar listener is available. Check startup errors.')
            history = capture.history
            rate = capture.rate
            hop = round(settings.hop * rate)
            window = round(settings.window * rate)
            last_scan = 0
            skipped = 0
            active = None
            last_endpoint = 0
            self.logger.info('Listening on %s (%s Hz) for %s.', capture.device_name, rate,
                             'Hey HAL or spacebar' if self.detector else 'spacebar')
            self.logger.info('Microphone input latency: %.3fs (requested %.3fs).',
                             getattr(capture, 'latency', settings.input_latency), settings.input_latency)
            self.logger.info('End-of-speech silence: %.2fs; query audio mode: %s.',
                             settings.silence, 'live' if audio_stream is not None else 'static')
            while True:
                capture.check()
                total, _ = history.position()
                pressed, released = keys.snapshot()
                if active is None and pressed is not None:
                    # Both key events are watched before inference starts, so a
                    # short press/release during decoding cannot strand recording.
                    start = max(0, history.sample_at(pressed) - round(.15 * rate))
                    active = {'kind': 'spacebar', 'start': start, 'detected': total}
                    on_trigger('spacebar')
                    self.logger.info('Push-to-talk triggered; record until spacebar release.')
                if active is not None:
                    elapsed = (total - active['start']) / rate
                    end = total
                    reason = None
                    if active['kind'] == 'spacebar' and released is not None:
                        if 'release_end' not in active:
                            active['release_end'] = history.sample_at(released) + round(.1 * rate)
                        end = min(total, active['release_end'])
                        # Keep a short tail after release; capture remains live.
                        if time.perf_counter() - released >= .1:
                            reason = 'spacebar released'
                    elif active['kind'] == 'wakeword' and total - last_endpoint >= round(.2 * rate):
                        last_endpoint = total
                        audio = to_audio(history.read(active['start'], total), rate)
                        speech_end = self.detector.last_speech_sample(audio)
                        speech_sample = (active['start'] + round(speech_end / RATE * rate)
                                         if speech_end is not None else None)
                        last_speech = (active['start'] + round(speech_end / RATE * rate)
                                       if speech_end is not None else active['detected'])
                        if (total - active['detected'] >= .5 * rate and
                                total - last_speech >= settings.silence * rate):
                            reason = 'silence'
                    if elapsed >= settings.maximum:
                        # Do not send a potentially incomplete instruction to the LLM.
                        raise CommandTooLongError(f'Command exceeded {settings.maximum:g} seconds; discarded to avoid a truncated request.')
                    if audio_stream is not None:
                        if stream_cursor is None:
                            audio_stream.start(rate)
                            stream_cursor = active['start']
                        if end < stream_cursor:
                            audio_stream.cancel('The live audio exceeded the final command boundary.')
                        elif reason or end - stream_cursor >= round(.1 * rate):
                            # Only copy native PCM here. Resampling and network IO
                            # run off-thread; each sample is queued exactly once.
                            if end > stream_cursor:
                                audio_stream.append(history.read(stream_cursor, end))
                                stream_cursor = end
                    if reason:
                        capture.check()
                        audio = to_audio(history.read(active['start'], end), rate)
                        if not len(audio):
                            raise RuntimeError('No microphone audio captured for this command.')
                        if audio_stream is not None:
                            audio_stream.end_audio()
                        if active['kind'] == 'spacebar' and self.detector is not None:
                            speech_end = self.detector.last_speech_sample(audio)
                            speech_sample = (active['start'] + round(speech_end / RATE * rate)
                                             if speech_end is not None else None)
                        if speech_sample is not None:
                            delivered_at = history.time_at(speech_sample)
                            if delivered_at is not None:
                                self.last_speech_end_at = delivered_at - getattr(
                                    capture, 'latency', settings.input_latency)
                        self.logger.info('Captured %.2fs via %s; endpoint: %s; skipped wake scan slots: %s.',
                                         len(audio) / RATE, active['kind'], reason, skipped)
                        completed = True
                        return audio, RATE
                    time.sleep(.02)
                    continue
                if self.detector is None or total - last_scan < hop:
                    time.sleep(.02)
                    continue
                skipped += max(0, (total - last_scan) // hop - 1)
                last_scan = total
                start = max(0, total - window)
                audio = to_audio(history.read(start, total), rate)
                if len(audio) < round(settings.window * RATE):
                    audio = np.pad(audio, (round(settings.window * RATE) - len(audio), 0))
                decision = self.detector.analyze(audio)
                # Reject a result if capture lost samples while inference ran.
                capture.check()
                # Spacebar has priority even if pressed during this inference.
                if keys.snapshot()[0] is not None:
                    continue
                if decision['matched']:
                    current, _ = history.position()
                    active = {'kind': 'wakeword', 'start': max(0, start - rate), 'detected': current}
                    on_trigger('wakeword')
                    self.logger.info('Wake detected by %s in %.3fs; skipped scan slots: %s.',
                                     ', '.join(decision['models']), decision['seconds'], skipped)
                    self.logger.debug('Wake transcript: %s', decision['transcripts'])
                elif decision['seconds'] > settings.hop:
                    self.logger.debug('Wake scan took %.3fs (interval %.3fs); next scan uses latest audio.',
                                      decision['seconds'], settings.hop)
        finally:
            if not completed:
                self.last_speech_end_at = None
                if audio_stream is not None:
                    audio_stream.cancel('The microphone command was discarded.')
            try:
                keys.close()
            finally:
                capture.close()
