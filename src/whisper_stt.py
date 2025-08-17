import os
import numpy as np
import whisper
import openai
from dotenv import load_dotenv
from speech_to_text import SpeechToText

load_dotenv()

TRANSCRIPTION_BACKEND = os.getenv("TRANSCRIPTION_BACKEND", "local").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "base")

class WhisperSTT(SpeechToText):
    def __init__(self, model_name=None):
        self.backend = TRANSCRIPTION_BACKEND
        self.model_name = model_name or WHISPER_MODEL_NAME
        self.model = None

        if self.backend == "local":
            self.model = whisper.load_model(self.model_name)

        elif self.backend == "api":
            if not OPENAI_API_KEY:
                raise RuntimeError("OPENAI_API_KEY not set but TRANSCRIPTION_BACKEND=api")
            self.client = openai.OpenAI(api_key=OPENAI_API_KEY)

        else:
            raise ValueError(f"Unknown TRANSCRIPTION_BACKEND: {self.backend}")

    def transcribe(self, audio_data, fs=16000):
        """
        audio_data: 1D numpy array float32, fs sample rate
        Returns: text transcription
        """

        if self.backend == "local":
            # Normalize
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val
            else:
                audio_data = np.zeros_like(audio_data)

            audio_data = whisper.pad_or_trim(audio_data)
            result = self.model.transcribe(audio_data, fp16=False)
            return result["text"]

        elif self.backend == "api":
            # Save temp wav for upload
            import tempfile, soundfile as sf

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, audio_data, fs, subtype="PCM_16")
                tmp_path = tmp.name

            with open(tmp_path, "rb") as f:
                transcript = self.client.audio.transcriptions.create(
                    model="gpt-4o-mini-transcribe",
                    file=f
                )

            os.remove(tmp_path)
            return transcript.text
