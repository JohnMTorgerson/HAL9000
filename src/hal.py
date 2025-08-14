import wave
import platform
import os
import sys
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from piper import PiperVoice, SynthesisConfig
from pydub import AudioSegment
from pydub.effects import normalize
from llm_client import LLMClient  # ensure llm_client.py is accessible
from whisper_stt import WhisperSTT  # import your external WhisperSTT
from weather_api import fetch_current_weather, fetch_weather_forecast
from wolfram_api import fetch_wolfram_answer  # import your external Wolfram API function
from news_api import fetch_top_headlines, fetch_articles_by_keyword

# Load environment variables from .env file
load_dotenv()

# ========== SET UP LOGGING ========== #
import logging
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

# Load HAL's voice
voice = PiperVoice.load("piper-models/hal.onnx")

# Configure synthesis with normal speech speed
syn_config = SynthesisConfig(
    volume=1.0,
    length_scale=1.0,  # normal speech speed
    noise_scale=1.0,
    noise_w_scale=1.0,
    normalize_audio=False
)

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

def add_reverb(input_wav, output_wav, delay_ms=120, decay=0.4, tail_volume_db=30):
    audio = AudioSegment.from_wav(input_wav)
    silence = AudioSegment.silent(duration=delay_ms)
    quieter_tail = audio - tail_volume_db
    delayed = silence + quieter_tail
    combined = audio.overlay(delayed, gain_during_overlay=-decay*10)
    combined = normalize(combined)
    combined.export(output_wav, format="wav")

def record_audio_interactive(fs=16000):
    import numpy as np

    logger.debug("Recording... Press Enter to stop.")
    recording = []

    def callback(indata, frames, time, status):
        if status:
            logger.debug(status)
        recording.append(indata.copy())

    stream = sd.InputStream(samplerate=fs, channels=1, callback=callback)
    stream.start()
    input()  # Wait for Enter to stop recording
    stream.stop()
    stream.close()

    audio = np.concatenate(recording).flatten()
    logger.debug("Recording complete.")
    return audio, fs

def get_user_input(stt):
    user_line = input("Type your question, or just press Enter to start recording (type 'exit' to quit): ").strip()
    if user_line.lower() == "exit":
        return None  # signal to quit

    if user_line == "":
        # Start recording immediately
        audio, fs = record_audio_interactive()
        text = stt.transcribe(audio, fs)
        logger.info(f"USER: {text}")
        return text
    else:
        return user_line


# # Example usage:
# city = "Minneapolis"
# logger.info(fetch_current_weather(city))
# logger.info(fetch_weather_forecast(city, days=3))


logger.info("========================= HAL 9000 is now online. Type 'exit' to shut me down.\n")

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

sst = WhisperSTT(model_name="base")

while True:
    user_input = get_user_input(sst)
    if user_input is None:
        logger.debug("HAL: Goodbye, Torgo.")
        break

    hal_reply = llm.get_response(user_input)

    if hal_reply.startswith("[EXTERNAL_API_CALL]"):
        logger.debug("HAL: Just a moment...")
        play_audio("../just_a_moment.aiff")  # Play "just a moment" clip to alert user that an API call is being made


        logger.info(f"HAL (external request): {hal_reply}")

        # Parse the API call command
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

        # logger.info(f"External data for HAL: {api_response}")

        # Send external data back to LLM for enriched response
        enriched_prompt = f"[EXTERNAL_API_RESPONSE] {api_response}"

        logger.info(f"Enriched prompt for HAL: {enriched_prompt}")

        hal_reply = llm.get_response(enriched_prompt)

    logger.info(f"HAL: {hal_reply}")

    with wave.open("hal_output.wav", "wb") as wav_file:
        voice.synthesize_wav(hal_reply, wav_file, syn_config=syn_config)

    # add_reverb("hal_output.wav", "hal_output_reverb.wav")
    play_audio("hal_output.wav")
