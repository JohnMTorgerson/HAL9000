"""Local Whisper wake decisions; query transcription stays in whisper_stt.py."""
import time
import numpy as np
import re
from wake_models import model_path

RATE = 16000
ACCEPTED_PAIRS = {('hey', word) for word in ('hal', 'hall', 'hell', 'how', 'al', 'howl')}

def matches_wake(text):
    words = re.findall(r'[a-z]+', text.casefold())
    return bool(set(zip(words, words[1:])) & ACCEPTED_PAIRS)

class SpeechActivityDetector:
    """Local speech boundaries without loading or running a Whisper decoder."""
    def __init__(self):
        from faster_whisper.vad import VadOptions, get_speech_timestamps
        self.get_speech_timestamps = get_speech_timestamps
        self.end_vad = VadOptions(threshold=.25, min_speech_duration_ms=100,
                                 min_silence_duration_ms=100, speech_pad_ms=0)

    def speech_bounds(self, audio):
        x = np.asarray(audio, dtype=np.float32)
        peak = float(np.max(np.abs(x))) if len(x) else 0
        if peak <= 1e-4:
            return None
        spans = self.get_speech_timestamps(x / peak * .95, self.end_vad)
        return (spans[0]['start'], spans[-1]['end']) if spans else None

    def last_speech_sample(self, audio):
        """Endpoint on the full buffered utterance, without trimming quiet words."""
        bounds = self.speech_bounds(audio)
        return bounds[1] if bounds else None


class WhisperWakeDetector(SpeechActivityDetector):
    def __init__(self, names=('base.en',), threads=2, max_gain_db=24,
                 normalization='capped', directory=None, beam_size=2):
        from faster_whisper import WhisperModel
        from faster_whisper.vad import VadOptions, get_speech_timestamps
        super().__init__()
        self.wake_vad = VadOptions(threshold=.25, min_speech_duration_ms=100,
                                  min_silence_duration_ms=400, speech_pad_ms=400)
        self.normalization = normalization
        self.max_gain_db = max_gain_db
        self.beam_size = beam_size
        self.models = {}
        for name in names:
            model = WhisperModel(str(model_path(name, directory)), device='cpu',
                                 compute_type='int8', cpu_threads=threads, local_files_only=True)
            list(model.transcribe(np.zeros(RATE, np.float32), language='en', beam_size=self.beam_size,
                 temperature=0, condition_on_previous_text=False, vad_filter=False)[0])
            self.models[name] = model
        get_speech_timestamps(np.zeros(RATE, np.float32), self.wake_vad)

    def analyze(self, audio):
        started = time.perf_counter()
        x = np.asarray(audio, dtype=np.float32)
        peak = float(np.max(np.abs(x))) if len(x) else 0
        peak_db = 20 * np.log10(max(peak, 1e-12))
        result = {'matched': False, 'models': [], 'transcripts': {}, 'seconds': 0., 'gain_db': 0.}
        if peak_db < -80:
            result['seconds'] = time.perf_counter() - started
            return result
        gain = min(self.max_gain_db, max(0, -3 - peak_db))
        if self.normalization == 'peak':
            gain = 20 * np.log10(.95 / peak)
        x = (x * 10**(gain / 20)).astype(np.float32)
        result['gain_db'] = float(gain)
        if self.get_speech_timestamps(x, self.wake_vad):
            for name, model in self.models.items():
                segments, _ = model.transcribe(x, language='en', beam_size=self.beam_size,
                    temperature=0, condition_on_previous_text=False, vad_filter=False,
                    initial_prompt=None, hotwords=None)
                parts = [s.text.strip() for s in segments]
                result['transcripts'][name] = ' '.join(parts)
                if any(matches_wake(text) for text in parts):
                    result['models'].append(name)
        result['matched'] = bool(result['models'])
        result['seconds'] = time.perf_counter() - started
        return result
