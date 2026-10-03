import io
import logging
import os

import numpy as np
import openai
from dotenv import load_dotenv
from speech_to_text import SpeechToText
from live_transcription import LiveSettings, LiveTranscription, TranscriptionError, NoSpeechError, _service_failure

load_dotenv()


class WhisperSTT(SpeechToText):
    API_MODEL = 'gpt-4o-mini-transcribe'

    def __init__(self, model_name=None, logger=None):
        self.logger = logger if logger is not None else logging.getLogger('HAL')
        self.backend = os.getenv('TRANSCRIPTION_BACKEND', 'local').strip().lower()
        self.mode = os.getenv('TRANSCRIPTION_MODE', 'static').strip().lower()
        self.model_name = model_name or os.getenv('WHISPER_MODEL_NAME', 'base')
        self.language = os.getenv('TRANSCRIPTION_LANGUAGE', 'en').strip().lower()
        self.model = None
        self.live_settings = None
        self.fallback = False
        if self.mode not in ('static', 'live'):
            raise ValueError('TRANSCRIPTION_MODE must be static or live.')
        if self.mode == 'live' and self.backend != 'api':
            raise ValueError('Live transcription requires TRANSCRIPTION_BACKEND=api. '
                             'Use TRANSCRIPTION_MODE=static for local transcription.')
        self.api_key = os.getenv('OPENAI_API_KEY')

        if self.backend == 'local':
            import whisper
            self.whisper = whisper
            self.model = whisper.load_model(self.model_name)
        elif self.backend == 'api':
            if not self.api_key:
                raise RuntimeError('OPENAI_API_KEY not set but TRANSCRIPTION_BACKEND=api')
            if self.mode == 'live':
                self.live_settings = LiveSettings.from_env()
                fallback = os.getenv('LIVE_TRANSCRIPTION_FALLBACK', 'false').strip().lower()
                if fallback not in ('true', 'false', '1', '0', 'yes', 'no', 'on', 'off'):
                    raise ValueError('LIVE_TRANSCRIPTION_FALLBACK must be true or false.')
                self.fallback = fallback in ('true', '1', 'yes', 'on')
                # Validate the optional dependency at startup, without connecting.
                LiveTranscription(self.api_key, self.live_settings, self.logger)
            self.client = openai.OpenAI(api_key=self.api_key, max_retries=0)
        else:
            raise ValueError(f'Unknown TRANSCRIPTION_BACKEND: {self.backend}')

    def create_live_stream(self):
        if self.mode == 'static':
            return None
        return LiveTranscription(self.api_key, self.live_settings, self.logger)

    def transcribe(self, audio_data, fs=16000, *, live_stream=None):
        """Return finalized text; partial live transcripts never become queries."""
        if self.mode == 'live':
            try:
                if live_stream is None:
                    raise TranscriptionError('Live transcription did not receive a streaming command.')
                text = live_stream.result()
                self.logger.info('Transcription source: live (%s).', self.live_settings.model)
                return text
            except NoSpeechError:
                # A completed empty result is not a service failure. Do not pay
                # for a fallback upload of the same noise/silence.
                raise
            except TranscriptionError as exc:
                if live_stream is not None:
                    live_stream.close()
                if not self.fallback:
                    raise
                self.logger.warning('%s Using the configured static API fallback; '
                                    'this may incur an additional transcription charge.', exc)
                return self._transcribe_static(audio_data, fs, source='static fallback')
        return self._transcribe_static(audio_data, fs)

    def _transcribe_static(self, audio_data, fs, source='static'):
        if self.backend == 'local':
            max_val = np.max(np.abs(audio_data))
            audio_data = audio_data / max_val if max_val > 0 else np.zeros_like(audio_data)
            audio_data = self.whisper.pad_or_trim(audio_data)
            text = self.model.transcribe(audio_data, fp16=False)['text']
        else:
            import soundfile as sf
            # A WAV in memory avoids temporary-file cleanup and keeps the
            # complete recording available for an explicitly enabled fallback.
            with io.BytesIO() as wav:
                sf.write(wav, audio_data, fs, format='WAV', subtype='PCM_16')
                wav.seek(0)
                try:
                    transcript = self.client.audio.transcriptions.create(
                        model=self.API_MODEL, file=('command.wav', wav, 'audio/wav'),
                        **({'language': self.language} if self.language else {}))
                except openai.APIError as exc:
                    raise _service_failure(exc) from None
                text = transcript.text
        if not isinstance(text, str):
            raise TranscriptionError('Invalid transcription response.')
        if not text.strip():
            raise NoSpeechError('Transcription returned no speech. Please repeat the request.')
        self.logger.info('Transcription source: %s (%s).', source,
                         self.model_name if self.backend == 'local' else self.API_MODEL)
        return text.strip()
