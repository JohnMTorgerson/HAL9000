import whisper
import sounddevice as sd
import numpy as np
import simpleaudio as sa

def record_audio(duration=5, fs=16000):
    print("Listening for your question... Speak now!")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    print("Recording complete.")
    return recording.flatten(), fs

def play_audio(audio_data, fs):
    # Normalize audio to 16-bit PCM for playback
    audio_int16 = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)
    play_obj = sa.play_buffer(audio_int16, 1, 2, fs)
    play_obj.wait_done()

def transcribe_audio(audio_data, model_name="base"):
    model = whisper.load_model(model_name)
    audio_float32 = audio_data.astype(np.float32)
    max_val = np.max(np.abs(audio_float32))
    if max_val > 0:
        audio_float32 /= max_val
    else:
        audio_float32 = np.zeros_like(audio_float32)

    audio_float32 = whisper.pad_or_trim(audio_float32)

    result = model.transcribe(audio_float32, fp16=False)
    return result["text"]

if __name__ == "__main__":
    audio, fs = record_audio()
    print("Playing back your recording...")
    play_audio(audio, fs)
    text = transcribe_audio(audio)
    print("You said:", text)
