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
import queue
import platform
SYSTEM = platform.system()
# ------------------ macOS Quartz fix for pynput ------------------ #
if SYSTEM == "Darwin":
    try:
        import Quartz
        # Force the constant to load
        _ = Quartz.CGEventGetIntegerValueField
        print("Quartz constant preloaded successfully.")
    except Exception as e:
        print(f"Failed to preload Quartz constants: {e}")


# ------------------------------------------------------------
# Load ENV
# ------------------------------------------------------------
load_dotenv()

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Recording Configuration
# ------------------------------------------------------------
RATE = 16000 # must be 16000 for porcupine
CHUNK_SIZE = 1024
PREBUFFER_DURATION = 0.8  # seconds of audio to keep before trigger
SILENCE_DURATION = 0.8 # seconds of silence to wait before stopping recording

# ------------------------------------------------------------
# Shared State for recording
# ------------------------------------------------------------
audio_queue = queue.Queue()
trigger_event = threading.Event()
trigger_type = {"value": None}
prebuffer = queue.deque(maxlen=int(PREBUFFER_DURATION * RATE / CHUNK_SIZE)) # ring buffer for prebuffering wake-word audio

# ------------------------------------------------------------
# Porcupine Wake Word Configuration
# ------------------------------------------------------------
ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY")
KEYWORDS = ["computer"]
KEYWORD_PATHS = os.getenv("KEYWORD_FILE_PATH")
porcupine = pvporcupine.create(access_key=ACCESS_KEY, keyword_paths=[KEYWORD_PATHS])

# ------------------------------------------------------------
# Load HAL voice 
# ------------------------------------------------------------
voice = PiperVoice.load("piper-models/hal.onnx")
syn_config = SynthesisConfig(volume=1.0, length_scale=1.0, noise_scale=1.0, noise_w_scale=1.0, normalize_audio=False)

# ------------------------------------------------------------
# Load Whisper – speech to text model
# ------------------------------------------------------------
stt = WhisperSTT(model_name="base")

# ------------------------------------------------------------
# LLM Configuration
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Main Loop
# ------------------------------------------------------------
def run():
    logger.info("========================= HAL 9000 is now online.\n")

    while True:
        try:
            # wait for trigger – either wake word or spacebar press
            trigger, stream, prebuffered_audio = wait_for_trigger()
            logger.info("====================================================================")

            # if spacebar, record until spacebar is released
            if trigger == "spacebar":
                logger.info("Push-to-talk (Spacebar hold) triggered.")
                audio, fs = record_while_spacebar_held(stream)
            # if wake word, record until silence threshold is met
            elif trigger == "wakeword":
                logger.info("Wake word triggered.")
                audio, fs = record_until_silence(stream, initial_audio=prebuffered_audio)
            # if neither, close stream and restart loop
            else:
                stream.stop()
                stream.close()
                continue

            # close stream after recording is finished
            stream.stop()
            stream.close()

            # normalize recorded audio
            audio = normalize_audio(audio)

            # save and play back command audio for debugging purposes
            sf.write("last_command.wav", audio, fs)
            logger.debug("Saved last command to last_command.wav – playing...")
            play_audio("last_command.wav")

            # transcribe audio to text
            user_input = stt.transcribe(audio, fs)
            logger.info(f"USER: {user_input}")

            # get HAL's response from LLM
            hal_reply = llm.get_response(user_input)

            # parse response and deal with any API calls
            if hal_reply.startswith("[EXTERNAL_API_CALL]"):
                logger.debug("HAL: Just a moment...")
                play_audio("HAL-clips/just_a_moment_normalized.aiff")

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

            # create audio from response text and save to file
            with wave.open("hal_output.wav", "wb") as wav_file:
                voice.synthesize_wav(hal_reply, wav_file, syn_config=syn_config)

            # normalize audio file
            audio, fs = sf.read("hal_output.wav", dtype="float32")
            normalized_audio = normalize_audio(audio)
            sf.write("hal_output.wav", normalized_audio, fs)

            # play audio of HAL's response from normalized file
            play_audio("hal_output.wav")

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Shutting down gracefully.")
            porcupine.delete()
            sys.exit(0)


# ------------------------------------------------------------
# Audio functions 
# ------------------------------------------------------------

def play_audio(file_path):
    if SYSTEM == "Darwin":
        os.system(f"afplay '{file_path}'")
    elif SYSTEM == "Windows":
        os.system(f"start \"\" \"{file_path}\"")
    elif SYSTEM == "Linux":
        output_device, device_fs = get_default_device("output")
        data, sr = sf.read(file_path, dtype="float32")
        if sr != device_fs:
            # Resample to device_fs
            data = np.interp(
                np.linspace(0, len(data), int(len(data) * device_fs / sr)),
                np.arange(len(data)),
                data
            ).astype(np.float32)
            sr = device_fs
        sd.play(data, samplerate=sr, device=output_device)
        sd.wait()
    else:
        logger.error(f"Cannot play audio automatically on {SYSTEM}. Please open {file_path} manually.")


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

# ----------------------------------------------------------------
# Helper function to get the correct audio device for input/output
# ----------------------------------------------------------------
def get_default_device(kind="input"):
    """
    Returns a tuple (device, samplerate) suitable for sounddevice streams.
    kind: "input" or "output"
    """
    devices = sd.query_devices()

    if kind == "input":
        if SYSTEM == "Linux":
            # pick USB mic if available
            for i, dev in enumerate(devices):
                if dev['max_input_channels'] > 0 and ("Microphone" in dev['name'] or "USB" in dev['name']):
                    return i, int(dev['default_samplerate'])
            # fallback: first input device
            for i, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    return i, int(dev['default_samplerate'])
        else:
            # macOS/Windows: use default device
            return None, RATE
    else:  # output
        if SYSTEM == "Linux":
            # pick USB speaker if available
            for i, dev in enumerate(devices):
                if dev['max_output_channels'] > 0 and ("USB" in dev['name'] or "Device" in dev['name']):
                    return i, int(dev['default_samplerate'])
            # fallback: first output device
            for i, dev in enumerate(devices):
                if dev['max_output_channels'] > 0:
                    return i, int(dev['default_samplerate'])
        else:
            # macOS/Windows: default output
            return None, 44100



# ------------------------------------------------------------
# Record until silence (used with wake word detection)
# ------------------------------------------------------------
def record_until_silence(stream, initial_audio=None, silence_threshold=0.001,
                         silence_duration=0.8, fs=RATE, max_duration=12.0):
    """
    Records audio until a period of silence is detected or max_duration is reached.
    Handles arbitrary device sample rates correctly.
    - initial_audio: numpy array of prebuffered audio (optional)
    - silence_threshold: RMS below which is considered silence
    - silence_duration: seconds of consecutive silence to stop recording
    - fs: target sample rate (default 16000)
    - max_duration: hard stop in seconds
    """
    recording = []

    # Include prebuffer if provided
    if initial_audio is not None:
        logger.debug(f"Initial prebuffer length: {len(initial_audio)} samples (~{len(initial_audio)/fs:.2f} sec)")
        recording.append(initial_audio.astype("float32"))

    # Determine device sample rate from stream
    device_fs = int(stream.samplerate)
    chunk_size = CHUNK_SIZE

    # Compute how many consecutive chunks equal desired silence duration
    chunk_duration_sec = chunk_size / fs
    max_silence_chunks = int(silence_duration / chunk_duration_sec)

    # Maximum chunks to prevent infinite recording
    max_chunks = int(max_duration / chunk_duration_sec)
    chunks_recorded = 0
    silence_counter = 0

    logger.info("Recording command (silence detection)...")
    start_time = time.time()

    while chunks_recorded < max_chunks:
        # Read a chunk from stream
        chunk, _ = stream.read(chunk_size)
        chunk = chunk.flatten().astype(np.float32) / 32768.0

        # Resample if device_fs != fs
        if device_fs != fs:
            chunk = np.interp(
                np.linspace(0, len(chunk), int(len(chunk) * fs / device_fs)),
                np.arange(len(chunk)),
                chunk
            ).astype(np.float32)

        recording.append(chunk)
        chunks_recorded += 1

        rms = np.sqrt(np.mean(chunk**2))
        if rms < silence_threshold:
            silence_counter += 1
        else:
            silence_counter = 0

        logger.debug(f"Chunk {chunks_recorded}: RMS={rms:.6f}, silence_counter={silence_counter}")

        if silence_counter >= max_silence_chunks:
            logger.debug(f"Silence threshold reached after {chunks_recorded} chunks.")
            break

    duration = time.time() - start_time
    audio = np.concatenate(recording)
    logger.info(f"Recording complete. Total duration: {len(audio)/fs:.2f} sec (loop time {duration:.2f} sec)")

    return audio, fs


# ------------------------------------------------------------
# Record while spacebar is held 
# ------------------------------------------------------------
def record_while_spacebar_held(stream, fs=RATE):
    """
    Records audio while the spacebar is held down.
    Stops immediately when the spacebar is released.
    """
    recording = []
    stop_event = threading.Event()
    device_fs = int(stream.samplerate)

    def on_release(key):
        if key == keyboard.Key.space:
            stop_event.set()
            return False

    listener = keyboard.Listener(on_release=on_release)
    listener.start()

    logger.info("Recording command (push-to-talk, hold spacebar)...")
    start_time = time.time()
    while not stop_event.is_set():
        chunk, _ = stream.read(CHUNK_SIZE)
        chunk = chunk.flatten().astype(np.float32) / 32768.0

        # Resample to 16kHz if needed
        if device_fs != fs:
            chunk = np.interp(
                np.linspace(0, len(chunk), int(len(chunk) * fs / device_fs)),
                np.arange(len(chunk)),
                chunk
            ).astype(np.float32)

        recording.append(chunk)

    listener.join()

    duration = time.time() - start_time
    audio = np.concatenate(recording) if recording else np.array([], dtype=np.float32)
    logger.info(f"Recording complete (spacebar released). Total duration: {len(audio)/fs:.2f} sec (loop time {duration:.2f} sec)")

    return audio, fs

# ------------------------------------------------------------
# Wait for trigger – either wake word or spacebar hold
# ------------------------------------------------------------
def wait_for_trigger(pre_buffer_duration=PREBUFFER_DURATION, fs=RATE):
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
    input_device, device_fs = get_default_device("input")

    # Start spacebar listener in a separate thread
    def spacebar_listener():
        def on_press(key):
            try:
                if key == keyboard.Key.space:
                    trigger_type["value"] = "spacebar"
                    trigger_event.set()
                    return False
            except Exception as e:
                logger.error(f"Spacebar listener exception: {e}")

        try:
            listener = keyboard.Listener(on_press=on_press)
            listener.daemon = True
            listener.start()
        except Exception as e:
            logger.error(f"pynput error: {e}")

    threading.Thread(target=spacebar_listener, daemon=True).start()

    # Set up prebuffer for wake word, using 16k for the frame rate (since we'll convert to that before extending the buffer)
    pre_buffer = deque(maxlen=int(pre_buffer_duration * fs))

    logger.info("Listening for wake word or push-to-talk (hold Spacebar)...")
    stream = sd.InputStream(samplerate=device_fs, channels=1, dtype="int16", device=input_device)
    stream.start()

    try:
        while not trigger_event.is_set():
            # porcupine expects 512 samples, so...
            # How many samples at device_fs give 512 samples at 16kHz
            device_frame_length = int(porcupine.frame_length * device_fs / fs)

            # Read that many samples from the device
            audio_frame, _ = stream.read(device_frame_length)
            audio_frame = audio_frame.flatten()

            # Now resample to exactly porcupine.frame_length
            if device_fs != fs:
                audio_16k = np.interp(
                    np.linspace(0, len(audio_frame), porcupine.frame_length),
                    np.arange(len(audio_frame)),
                    audio_frame
                ).astype(np.int16)
            else:
                audio_16k = audio_frame

            # extend prebuffer with converted audio frame
            pre_buffer.extend(audio_16k)

            keyword_index = porcupine.process(audio_16k)
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

# ------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------
if __name__ == "__main__":
    run()