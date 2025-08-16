import whisper
import numpy as np
from speech_to_text import SpeechToText

class WhisperSTT(SpeechToText):
    def __init__(self, model_name="base"):
        self.model = whisper.load_model(model_name)

    def transcribe(self, audio_data, fs=16000):
        # audio_data: 1D numpy array float32, fs sample rate
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
        else:
            audio_data = np.zeros_like(audio_data)

        audio_data = whisper.pad_or_trim(audio_data)
        result = self.model.transcribe(audio_data, fp16=False)
        return result["text"]
