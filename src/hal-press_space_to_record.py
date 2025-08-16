import wave
import platform
import os
import sys
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from piper import PiperVoice, SynthesisConfig
from pydub import AudioSegment
from pydub.effects import normalize
from llm_client import LLMClient
from whisper_stt import WhisperSTT
from weather_api import fetch_current_weather, fetch_weather_forecast
from wolfram_api import fetch_wolfram_answer
from news_api import fetch_top_headlines, fetch_articles_by_keyword
import pvporcupine
import logging
from collections import deque
from pynput import keyboard
import threading

# ------------------ Load env and logging ------------------ #
load_dotenv()

logger = logging.getLogger('HAL')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s %(name)s.%(funcName)s() line %(lineno)s %(levelname).5s :: %(message)s")
# log to file at INFO level
file_handler = logging.FileHandler(os.path.abspath(f"{os.environ['LOG_PATH']}/log.log"))
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
# log to another file at ERROR level
error_file_handler = logging.FileHandler(os.path.abspath(f"{os.environ['LOG_PATH']}/error.log"))
error_file_handler.setLevel(logging.ERROR)
error_file_handler.setFormatter(formatter)
logger.addHandler(error_file_handler)
# log to console at DEBUG level
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# ------------------ Load HAL voice ------------------ #
voice = PiperVoice.load("piper-models/hal.onnx")
syn_config = SynthesisConfig(volume=1.0, length_scale=1.0, noise_scale=1.0, noise_w_scale=1.0, normalize_audio=False)

def play_audio(file_path):
    system = platform.system()
    if system == "Darwin":
        os.system(f"afplay {file_path}")
    elif system == "Windows":
        os.system(f"start {file_path}")
    elif system == "Linux":
        os.system(f"aplay {file_path}")
    else:
        logger.error(f"Cannot play audio automatically on {system}. Please open {file_path} manually.")

def normalize_audio(audio, peak=0.95):
    """
    Normalize a float32 audio array to the given peak amplitude.
    """
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = (audio / max_val) * peak
    return audio


def add_reverb(input_wav, output_wav, delay_ms=120, decay=0.4, tail_volume_db=30):
    audio = AudioSegment.from_wav(input_wav)
    silence = AudioSegment.silent(duration=delay_ms)
    quieter_tail = audio - tail_volume_db
    delayed = silence + quieter_tail
    combined = audio.overlay(delayed, gain_during_overlay=-decay*10)
    combined = normalize(combined)
    combined.export(output_wav, format="wav")

# ------------------ Whisper + Silence detection ------------------ #
sst = WhisperSTT(model_name="base")

def record_until_silence(stream, initial_audio=None, silence_threshold=0.001, silence_duration=0.8, fs=16000, max_duration=12.0):
    """
    Records audio from the mic until a period of silence is detected
    or until max_duration (seconds) is reached.
    - initial_audio: numpy array of prebuffered audio (optional, from wakeword detection)
    - silence_threshold: RMS below which is considered silence
    - silence_duration: seconds of consecutive silence to stop recording
    - max_duration: maximum recording time in seconds
    """
    chunk_size = 1024
    recording = []

    if initial_audio is not None:
        logger.debug(f"Initial prebuffer length: {len(initial_audio)} samples (~{len(initial_audio)/fs:.2f} sec)")
        recording.append(initial_audio.astype("float32"))

    silence_counter = 0
    max_silence_chunks = int(silence_duration * fs / chunk_size)
    max_chunks = int(max_duration * fs / chunk_size)
    chunks_recorded = 0

    logger.info("Recording command...")
    start_time = time.time()
    while chunks_recorded < max_chunks:
        chunk, _ = stream.read(chunk_size)
        chunk = chunk.flatten().astype(np.float32) / 32768.0  # convert int16 -> float32 -1.0..1.0
        recording.append(chunk)
        chunks_recorded += 1

        rms = np.sqrt(np.mean(chunk**2))
        if rms < silence_threshold:
            silence_counter += 1
        else:
            silence_counter = 0

        # Debug logging for timing and RMS
        logger.debug(f"Chunk {chunks_recorded}: RMS={rms:.6f}, silence_counter={silence_counter}")

        if silence_counter > max_silence_chunks:
            logger.debug(f"Silence threshold reached after {chunks_recorded} chunks.")
            break

    duration = time.time() - start_time
    audio = np.concatenate(recording)
    logger.info(f"Recording complete. Total duration: {len(audio)/fs:.2f} sec (loop time {duration:.2f} sec)")
    return audio, fs

# ------------------ Porcupine Wake Word ------------------ #
ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY")
KEYWORDS = ["computer"]
KEYWORD_PATHS = os.getenv("KEYWORD_FILE_PATH")
porcupine = pvporcupine.create(access_key=ACCESS_KEY, keyword_paths=[KEYWORD_PATHS])

# ------------------ Wait for trigger ------------------ #
# This function listens for either a wake word or a Spacebar key press to start recording.
# If Spacebar is pressed, it starts recording immediately without waiting for the wake word.
def wait_for_trigger(pre_buffer_duration=0.8, fs=16000):
    """
    Waits for either the wake word or the Spacebar key to trigger recording.
    Returns (trigger_type, stream, buffered_audio)
    - trigger_type: 'wakeword' or 'spacebar'
    - stream: the active InputStream (to reuse)
    - buffered_audio: prebuffered audio if wakeword triggered, else None
    """
    trigger_event = threading.Event()
    trigger_type = {"value": None}
    buffered_audio_container = {"audio": None}

    def spacebar_listener():
        """Sets trigger_event when spacebar is pressed."""
        def on_press(key):
            try:
                if key == keyboard.Key.space:
                    trigger_type["value"] = "spacebar"
                    trigger_event.set()
                    return False  # stop listener
            except Exception as e:
                logger.error(f"Spacebar listener exception: {e}")

        try:
            listener = keyboard.Listener(on_press=on_press)
            listener.daemon = True
            listener.start()
        except Exception as e:
            logger.error(f"pynput error: {e}")

    threading.Thread(target=spacebar_listener, daemon=True).start()

    frame_length = porcupine.frame_length
    buffer_size = int(pre_buffer_duration * fs)
    pre_buffer = deque(maxlen=buffer_size)

    logger.info("Listening for wake word or push-to-talk (Spacebar)...")
    stream = sd.InputStream(samplerate=fs, channels=1, dtype="int16")
    stream.start()
    try:
        while not trigger_event.is_set():
            audio_frame, _ = stream.read(frame_length)
            audio_frame = audio_frame.flatten()
            pre_buffer.extend(audio_frame)

            keyword_index = porcupine.process(audio_frame)
            if keyword_index >= 0:
                trigger_type["value"] = "wakeword"
                buffered_audio_container["audio"] = np.array(pre_buffer, dtype=np.float32) / 32768.0
                trigger_event.set()
                break
    except KeyboardInterrupt:
        stream.stop()
        stream.close()
        raise

    return trigger_type["value"], stream, buffered_audio_container["audio"]


# ------------------ LLM Setup ------------------ #
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = LLMClient(
    backend="openai",
    model_name="gpt-4.1-nano",
    max_history=10,
    openai_api_key=openai_api_key
)

# llm = LLMClient(
#     backend="ollama",
#     model_name="llama3",  # or whichever Ollama model you want to use
#     max_history=10
# )

# ------------------ Main loop ------------------ #
logger.info("========================= HAL 9000 is now online.\n")

while True:
    try:
        # Listen for wake word or Spacebar key press
        trigger, stream, prebuffered_audio = wait_for_trigger()

        logger.info("====================================================================")

        if trigger == "spacebar":
            logger.info("Push-to-talk (Spacebar) triggered.")
            audio, fs = record_until_silence(stream)  # reuse same stream
        elif trigger == "wakeword":
            logger.info("Wake word triggered.")
            audio, fs = record_until_silence(stream, initial_audio=prebuffered_audio) # reuse same stream
        else:
            stream.stop()
            stream.close()
            continue

        # close stream after recording
        stream.stop()
        stream.close()

        # Normalize the recorded command so it’s not too quiet
        audio = normalize_audio(audio)

        # Save user command to file and play it back for debugging
        sf.write("last_command.wav", audio, fs)
        logger.debug("Saved last command to last_command.wav – playing...")
        play_audio("last_command.wav")

        user_input = sst.transcribe(audio, fs)
        logger.info(f"USER: {user_input}")

        hal_reply = llm.get_response(user_input)

        if hal_reply.startswith("[EXTERNAL_API_CALL]"):
            logger.debug("HAL: Just a moment...")
            play_audio("HAL-clips/just_a_moment.aiff") # Play a short clip to alert user that an external API call is being made

            logger.info(f"HAL (external request): {hal_reply}")
            command = hal_reply[len("[EXTERNAL_API_CALL]"):].strip().split()
            api_type = command[0].lower()
            params = command[1:]

            if api_type == "weather":
                city = " ".join(params)
                api_response = fetch_current_weather(city)
            elif api_type == "forecast":
                try:
                    days = int(params[-1])
                    city = " ".join(params[:-1])
                except ValueError:
                    days = 1
                    city = " ".join(params)
                api_response = fetch_weather_forecast(city, days=days)
            elif api_type == "wolfram":
                query = " ".join(params)
                api_response = fetch_wolfram_answer(query)
            elif api_type == "news":
                if params:
                    keyword = " ".join(params)
                    api_response = fetch_articles_by_keyword(keyword)
                else:
                    api_response = fetch_top_headlines()
            else:
                api_response = f"Unknown API request type: {api_type}"

            enriched_prompt = f"[EXTERNAL_API_RESPONSE] {api_response}"
            logger.info(f"Enriched prompt for HAL: {enriched_prompt}")
            hal_reply = llm.get_response(enriched_prompt)

        logger.info(f"HAL: {hal_reply}")

        with wave.open("hal_output.wav", "wb") as wav_file:
            voice.synthesize_wav(hal_reply, wav_file, syn_config=syn_config)

        play_audio("hal_output.wav")

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down gracefully.")
        porcupine.delete()
        sys.exit(0)
