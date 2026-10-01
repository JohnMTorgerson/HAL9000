import wave
import platform
import os
import sys
import subprocess
import time
import numpy as np
from scipy.signal import resample_poly
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from piper import PiperVoice, SynthesisConfig
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range
import io
from llm_client import LLMClient, LLMServiceError
from audio_devices import choose_input_device
from whisper_stt import WhisperSTT
from live_transcription import TranscriptionError
from followup import FollowupSettings, FollowupSession, explicitly_addresses_hal
from voice_input import VoiceInput, CommandTooLongError
from audio_capture import AudioOverflowError
from weather_api import fetch_current_weather, fetch_weather_forecast
from wolfram_api import fetch_wolfram_answer
from news_api import fetch_top_headlines, fetch_articles_by_keyword
from calendar_api import ICloudCalendar
calendar_backend = ICloudCalendar()
from sports_api import SportsRouter
sports_backend = SportsRouter()
from maps_api import MapsRouter
maps_backend = MapsRouter()
import logging
from display_log_handler import DisplayPushHandler
import platform
from led_manager import get_led
import json
import re
import shlex
from helper_funcs import looks_factual, extract_named_entities, strip_name_at_sentence_end
from display.server_lifecycle import DisplayServerManager
from display_client import DisplayClient
display = DisplayClient(os.getenv("DISPLAY_SERVER_URL", "http://127.0.0.1:8000"))


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
# ---- Custom "DISPLAY" log level (between INFO and WARNING) ----
DISPLAY_LEVEL = 25
logging.addLevelName(DISPLAY_LEVEL, "DISPLAY")
def log_display(self, msg, *args, **kwargs):
    if self.isEnabledFor(DISPLAY_LEVEL):
        self._log(DISPLAY_LEVEL, msg, args, **kwargs)
logging.Logger.display = log_display  # e.g., logger.display("…")

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

# STREAM CLEAN LOGS TO THE DISPLAY (bottom panel)
display_handler = DisplayPushHandler(
    base_url=os.getenv("DISPLAY_SERVER_URL", "http://127.0.0.1:8000"),
    slots=("bottom",),          # or ("top",) if you prefer
    priority=70,                # below maps/etc. but above slideshow
    key="logs",                 # stable key so we upsert + refresh TTL
    ttl_secs=30,                # auto-hide ~30s after last update
    max_lines=80,
    max_chars=4000,
    min_push_interval=0.25,
    timeout=0.8,
)
display_handler.setLevel(DISPLAY_LEVEL) # Only show DISPLAY and above on the screen
display_handler.setFormatter(logging.Formatter("%(asctime)s :: %(message)s", datefmt="%H:%M:%S")) # Minimal on-screen format
logger.addHandler(display_handler)


# ------------------------------------------------------------
# Misc
# ------------------------------------------------------------
DEBUG_ON = os.getenv("DEBUG_ON") == "True"
DEBUG_PLAYBACK = os.getenv("DEBUG_PLAYBACK", "false").strip().lower() in ("1", "true", "yes", "on")
PLATFORM = os.getenv("PLATFORM")
USER = os.getenv("HAL_USER_NAME", "Dave").capitalize() # Default to "Dave" if HAL_USER_NAME not set in environment

# ------------------------------------------------------------
# Recording/Playback Configuration
# ------------------------------------------------------------
RATE = 16000 # query audio sample rate
COMPRESSION_THRESHOLD = float(os.getenv("COMPRESSION_THRESHOLD",0)) # amount to compress audio before playing back
HI_PASS_FREQ = int(os.getenv("HI_PASS_FREQ",0))
# if PLATFORM == "pi":
#     sd.default.device = "pulse"

# ------------------------------------------------------------
# Load HAL voice 
# ------------------------------------------------------------
voice = PiperVoice.load("piper-models/hal.onnx")
syn_config = SynthesisConfig(volume=1.0, length_scale=1.0, noise_scale=1.0, noise_w_scale=1.0, normalize_audio=False)

# ------------------------------------------------------------
# Load Whisper – speech to text model
# ------------------------------------------------------------
stt = WhisperSTT(logger=logger)
logger.info('Query transcription ready: %s / %s (%s).', stt.mode, stt.backend,
            stt.live_settings.model if stt.mode == 'live' else
            stt.model_name if stt.backend == 'local' else stt.API_MODEL)
if stt.mode == 'live':
    logger.info('Live transcription delay: %s; static API fallback: %s.',
                stt.live_settings.delay, 'enabled' if stt.fallback else 'disabled')
# The detector is local and lightweight; query transcription still uses stt above.
followup_settings = FollowupSettings.from_env()
voice_input = VoiceInput.from_env(logger, device_selector=lambda: get_default_device("input")[0],
                                 followup_enabled=followup_settings.enabled)

# ------------------------------------------------------------
# LLM Configuration
# ------------------------------------------------------------
LLM_BACKEND = os.getenv("LLM_BACKEND", "openai")
if LLM_BACKEND == "openai":
    llm = LLMClient(
        backend="openai",
        model_name=os.getenv("LLM_MODEL"),
        max_history=int(os.getenv("LLM_MAX_HISTORY")),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        service_tier=os.getenv("LLM_SERVICE_TIER"),
        logger=logger,
    )
elif LLM_BACKEND == "ollama":
    llm = LLMClient(
        backend="ollama",
        model_name=os.getenv("LLM_MODEL"),
        max_history=int(os.getenv("LLM_MAX_HISTORY"))
    )
else:
    llm = None
    raise ValueError(f"Unknown LLM Backend: {LLM_BACKEND}")

if followup_settings.enabled and LLM_BACKEND != 'openai':
    raise ValueError('FOLLOWUP_ENABLED requires LLM_BACKEND=openai with structured-output support.')
logger.info('Follow-up listening: %s; window %.1fs; session limit %.1fs.',
            'enabled' if followup_settings.enabled else 'disabled',
            followup_settings.window, followup_settings.session_limit)

# ------------------------------------------------------------
# Get LED if on raspberry pi, dummy if not
# ------------------------------------------------------------
led = get_led()

# ------------------------------------------------------------
# Main Loop
# ------------------------------------------------------------
def run():
    logger.info("========================= HAL 9000 is now online.\n")
    followups = FollowupSession(followup_settings, clock=time.perf_counter)

    # start or connect to display server
    display_mgr = DisplayServerManager(
        url=os.getenv("DISPLAY_SERVER_URL", "http://127.0.0.1:8000"),
        logger=logger,
    )
    display_mgr.start()

    while True:
        live_stream = None
        try:
            triggered_at = None
            trigger_kind = None
            # Capture continues while the wake detector processes its rolling window.
            # Live transcription can overlap capture. The microphone still
            # closes before any query reaches the LLM or HAL starts speaking.
            def on_trigger(kind):
                nonlocal triggered_at, trigger_kind
                triggered_at = time.perf_counter()
                trigger_kind = kind
                logger.info("====================================================================")
                if kind == 'followup':
                    logger.info('Possible follow-up speech: lighting LED; intent not yet confirmed.')
                else:
                    logger.info("Detected %s command: lighting LED", kind)
                led.on()

            capture_options = {'on_trigger': on_trigger}
            deadline = followups.deadline()
            if deadline is not None:
                capture_options['followup_deadline'] = deadline
            if stt.mode == 'live':
                live_stream = stt.create_live_stream()
                capture_options['audio_stream'] = live_stream
            captured = voice_input.read_command(**capture_options)
            if captured is None:
                followups.close()
                continue
            audio, fs = captured
            capture_ready_at = time.perf_counter()
            speech_ended_at = voice_input.last_speech_end_at
            if triggered_at is not None:
                logger.info('Timing: trigger to capture ready %.3fs (includes microphone cleanup).',
                            capture_ready_at - triggered_at)
            if speech_ended_at is not None:
                logger.info('Timing: estimated speech end to capture ready %.3fs.',
                            capture_ready_at - speech_ended_at)

            # normalize recorded audio
            stage_started = time.perf_counter()
            audio = normalize_audio(audio)

            # Keep the debug recording; replay requires its own explicit opt-in.
            if DEBUG_ON or DEBUG_PLAYBACK:
                sf.write("last_command.wav", audio, fs)
                logger.debug("Saved last command to last_command.wav")
            logger.info('Timing: query audio preparation %.3fs.', time.perf_counter() - stage_started)
            if DEBUG_PLAYBACK and trigger_kind != 'followup':
                logger.debug("DEBUG_PLAYBACK enabled – playing last command...")
                play_audio("last_command.wav", label='debug query')

            # transcribe audio to text
            stage_started = time.perf_counter()
            if live_stream is not None:
                user_input = stt.transcribe(audio, fs, live_stream=live_stream)
                logger.info('Timing: transcription wait after capture %.3fs (live or configured fallback).',
                            time.perf_counter() - stage_started)
            else:
                user_input = stt.transcribe(audio, fs)
                logger.info('Timing: query transcription %.3fs.', time.perf_counter() - stage_started)
            # get HAL's response from LLM
            stage_started = time.perf_counter()
            explicit = trigger_kind in ('wakeword', 'spacebar')
            if trigger_kind == 'followup':
                explicit = explicitly_addresses_hal(user_input)
                decision = llm.get_followup_response(user_input, explicitly_addressed=explicit)
                logger.info('Timing: follow-up decision and response %.3fs; decision=%s.',
                            time.perf_counter() - stage_started, decision.decision)
                if decision.decision != 'respond':
                    led.off()
                    if decision.decision == 'end':
                        followups.close()
                        logger.info('Follow-up session ended; wake phrase or spacebar required again.')
                    else:
                        logger.info('Follow-up ignored; the existing deadline is unchanged.')
                    continue
                hal_reply = decision.reply
                logger.display(f"USER: {user_input}")
            else:
                logger.display(f"USER: {user_input}")
                stage_started = time.perf_counter()
                hal_reply = llm.get_response(user_input)
            logger.info('Timing: initial LLM response %.3fs.', time.perf_counter() - stage_started)

            # If HAL claims not to know, force it to try Wikipedia before giving up
            # first testing if the query looks like a factual question about a named entity we can search for
            if re.search(r"(i\s+don.?t\s+know|i\s+don.?t\s+have|i.?m\s+sorry.*can.?t\s+do)", hal_reply.strip(), re.I):
                named_entities = extract_named_entities(user_input)
                if DEBUG_ON:
                    logger.debug(f"HAL responded with ignorance: {hal_reply}")
                    logger.debug(f"Searching for named entities in query...")
                    logger.debug(f"Named entities: {named_entities}")
                if looks_factual(user_input) and named_entities:
                    logger.warning("HAL ignorance detected on factual question – forcing Wikipedia search")
                    hal_reply = f"[EXTERNAL_API_CALL] wikipedia search {named_entities[0]}"
                else:
                    logger.debug("Either no named entities found or question was not parsed as factual. NOT forcing wikipedia search")

            # keep handling API calls until HAL gives a final answer
            while hal_reply.startswith("[EXTERNAL_API_CALL]"):
                logger.display("HAL: Just a moment...")
                play_audio("HAL-clips/just_a_moment_normalized.aiff", label='acknowledgment',
                           triggered_at=triggered_at, capture_ready_at=capture_ready_at,
                           speech_ended_at=speech_ended_at)

                logger.display(f"HAL (external request): {hal_reply}")
                command = shlex.split(hal_reply[len("[EXTERNAL_API_CALL]"):].strip()) # shlex splits by space, except respect quotes
                api_type = command[0].lower()
                params = command[1:]

                stage_started = time.perf_counter()
                api_response = handle_api_call(api_type, params, user_input)
                logger.info('Timing: external request %s %.3fs.', api_type, time.perf_counter() - stage_started)
                enriched_prompt = f"[EXTERNAL_API_RESPONSE] {api_response}"

                if not DEBUG_ON and len(enriched_prompt) > 800:
                    enriched_prompt = enriched_prompt[:800] + "[...]\n[TRUNCATED (for logging only)]"
                logger.info(f"Enriched prompt for HAL: {enriched_prompt}")


                stage_started = time.perf_counter()
                hal_reply = llm.get_response(enriched_prompt)
                logger.info('Timing: follow-up LLM response %.3fs.', time.perf_counter() - stage_started)

            # sanitize HAL's habit of ending sentences with ", {USER}"
            filtered_reply = strip_name_at_sentence_end(hal_reply, name=USER)
            if DEBUG_ON and filtered_reply != hal_reply:
                logger.debug(f"Post-processed HAL reply:\nBEFORE: {hal_reply}\nAFTER : {filtered_reply}")
            hal_reply = filtered_reply

            logger.display(f"HAL: {hal_reply}")

            # create audio from response text and save to file
            stage_started = time.perf_counter()
            with wave.open("hal_output.wav", "wb") as wav_file:
                voice.synthesize_wav(hal_reply, wav_file, syn_config=syn_config)
            logger.info('Timing: voice synthesis %.3fs.', time.perf_counter() - stage_started)

            # normalize audio file
            stage_started = time.perf_counter()
            audio, fs = sf.read("hal_output.wav", dtype="float32")
            normalized_audio = normalize_audio(audio)
            sf.write("hal_output.wav", normalized_audio, fs)
            logger.info('Timing: reply audio normalization %.3fs.', time.perf_counter() - stage_started)

            # play audio of HAL's response from normalized file
            play_audio("hal_output.wav", label='reply', triggered_at=triggered_at,
                       capture_ready_at=capture_ready_at, speech_ended_at=speech_ended_at)
            # Never open a window after "Just a moment" or while HAL speaks.
            followups.after_response(explicit=explicit)

            #turn LED off
            logger.info("Turning LED off")
            led.off()

        except AudioOverflowError as exc:
            followups.close()
            led.off()
            logger.warning('%s Reopening microphone.', exc)
            logger.display('HAL: Microphone audio was lost. Please repeat your request when listening resumes.')
            time.sleep(.5)
            continue
        except CommandTooLongError as exc:
            followups.close()
            logger.warning("%s", exc)
            logger.display("That request was too long. Please try a shorter request.")
            led.off()
            continue

        except (LLMServiceError, TranscriptionError) as exc:
            followups.close()
            led.off()
            logger.error("%s", exc)
            logger.display(f"HAL: {exc}")
            logger.info("Returning to listening.")
            continue

        except KeyboardInterrupt:
            followups.close()
            logger.info("Keyboard interrupt received. Shutting down gracefully.")
            display_mgr.stop()
            led.off()
            sys.exit(0)

        except Exception:
            followups.close()
            display_mgr.stop()
            led.off()
            raise
        finally:
            if live_stream is not None:
                live_stream.close()

# ------------------------------------------------------------
# API CALL
# ------------------------------------------------------------
def handle_api_call(api_type, params, user_input):
    """
    Executes an external API call based on type and params.
    Returns a string `api_response` that will be fed back to HAL.
    """
    try:
        if api_type == "weather":
            city = " ".join(params)
            return fetch_current_weather(city)

        elif api_type == "forecast":
            try:
                days = int(params[-1])
                city = " ".join(params[:-1])
            except ValueError:
                days = 1
                city = " ".join(params)
            return fetch_weather_forecast(city, days=days)

        elif api_type == "wolfram":
            query = " ".join(params)
            return fetch_wolfram_answer(query)

        elif api_type == "news":
            if params:
                keyword = " ".join(params)
                return fetch_articles_by_keyword(keyword)
            else:
                return fetch_top_headlines()

        elif api_type == "wikipedia":
            subcommand = params[0].lower()
            if subcommand == "search":
                query = " ".join(params[1:])
                from wikipedia_api import search_wikipedia
                results = search_wikipedia(query)
                if results:
                    return (
                        f"Wikipedia search results for '{query}':\n"
                        f"{json.dumps(results)}\n\n"
                        "DO NOT attempt to answer the user's question based on the above information. "
                        "DO NOT give up on answering the question. Pick the most relevant article from the JSON search results, "
                        "and respond with another API call as instructed, in order to receive either a summary of the article or "
                        "the full text of the article. ONLY THEN may you respond to the user."
                    )
                else:
                    return f"No Wikipedia results found for '{query}'."

            elif subcommand == "fetch":
                mode = params[1].lower() if len(params) > 2 else "summary"
                title = " ".join(params[2:]).strip('"')  # strip quotes if HAL added them
                from wikipedia_api import fetch_wikipedia
                page = fetch_wikipedia(title, mode=mode)
                helper_prompt = (
                    f"Use the following Wikipedia article to answer the user's query: '{user_input}'.\n"
                    f"- Do not summarize the entire article unless explicitly asked.\n"
                    f"- Do not say 'I'm sorry {USER}. I'm afraid I can't do that.'\n"
                    f"- Answer directly based on the text."
                )
                if mode == "summary":
                    return f"{helper_prompt}\n\n[ARTICLE START]\n{page['title']} (summary): {page.get('extract','')}\nURL: {page.get('url','')}\n[ARTICLE END]"
                else:
                    return f"{helper_prompt}\n\n[ARTICLE START]\n{page['title']} (full article):\n{page.get('text','')}\n[ARTICLE END]"

            else:
                return f"Unknown Wikipedia subcommand: {subcommand}"
            
        elif api_type.startswith("calendar"):
            subcommand = api_type # for calendar requests, the api_type is also the command: e.g. calendar_search
            response = calendar_backend.dispatch(subcommand,params)
            if isinstance(response, dict) and "error" in response:
                return json.dumps(response)

            # push a calendar overlay to the display
            # suppose `response` is a list of event dicts from your calendar backend
            # each item can have: title, start, end, location (strings), open_now (bool)
            display.calendar(
                response,
                tz=os.getenv("TIMEZONE", "America/Chicago"),
                on=("top",),
                priority=80,
                ttl=90,                 # hide after ~90s of inactivity
                key="calendar",
                title="SCHEDULE",
                code="OPS 12–A",
                accent="#1DE9D6",       # optional
                limit=12                # optional
            )

            return json.dumps(response)

        elif api_type == "sports":
            if not params or len(params) < 2:
                return json.dumps({"error": "Missing sports command or params"})

            command = params[0]  # next_game, schedule, standings
            team_or_league = params[1]
            if len(params) > 2:
                team2 = params[2]
                response = sports_backend.dispatch(command, team_or_league, team2)
            else:
                response = sports_backend.dispatch(command, team_or_league)
            return json.dumps(response)
        
        elif api_type == "maps":
            if not params or len(params) < 2:
                return json.dumps({"error": "Missing maps command or params"})

            command = params[0]  # for now just "search"
            maps_params = {"query": params[1]}
            if len(params) >= 3:
                maps_params["radius"] = params[2]

            response = maps_backend.dispatch(command, maps_params)

            # Attempt to extract a list of places from 'response'
            try:
                res_obj = response if isinstance(response, (list, dict)) else json.loads(response)
            except Exception:
                res_obj = response

            if isinstance(res_obj, list):
                places = res_obj
            elif isinstance(res_obj, dict):
                places = res_obj.get("results") or res_obj.get("items") or res_obj.get("places") or []
            else:
                places = []

            # If any place has coordinates, show a map on the top panel
            has_coords = any(
                (p.get("lat") or p.get("latitude") or (p.get("geometry", {}).get("location", {}).get("lat") if isinstance(p, dict) else None)) is not None
                and
                (p.get("lon") or p.get("lng") or p.get("longitude") or (p.get("geometry", {}).get("location", {}).get("lng") if isinstance(p, dict) else None)) is not None
                for p in places
            )
            if has_coords:
                try:
                    display.map(places, on=("top",), priority=80, ttl=120, key="map", fullscreen=False, scale=4, zoom=14)
                except Exception as e:
                    logger.warning(f"Failed to push map overlay: {e}")
            else:
                logger.debug("Maps response missing lat/lon; skipping map overlay.")

            return json.dumps(response)


        else:
            return f"Unknown API request type: {api_type}"

    except Exception as e:
        logger.error(f"{api_type} API call failed: {e}")
        return f"{api_type} API error: {e}"





# ------------------------------------------------------------
# Audio functions 
# ------------------------------------------------------------

# def play_audio(filename, threshold_dB=COMPRESSION_THRESHOLD):
#     """
#     Plays an audio file with optional gain boost and soft limiting to prevent clipping.
#     
#     Parameters:
#         filename: path to WAV file
#         threshold_dB: peak threshold for limiting (dBFS)
#     """
#     # Load audio
#     audio = AudioSegment.from_file(filename, format="wav")
#     
#     # Normalize to -1 dBFS 
#     logger.debug(f"Normalizing {filename}")
#     audio = normalize(audio)
#         
#     if threshold_dB < 0:
#         # Apply limiter
#         logger.debug(f"Compressing {filename} at {threshold_dB}dB threshold")
#         audio = compress_dynamic_range(
#             audio,
#             threshold=threshold_dB,
#             ratio=100.0,
#             attack=5,
#             release=5
#         )
#
#         # Boost overall gain
#         logger.debug(f"Boosting gain for {filename}")
#         audio = audio - threshold_dB * 0.8 # I'm doing this because renormalizing wasn't working
#
#         # logger.debug(f"Renormalizing {filename}")
#         # audio = normalize(audio)
#
#     else:
#         logger.debug(f"Compression threshold is {threshold_dB}, not compressing")
#     
#     # Export to raw data for playback
#     raw_audio = io.BytesIO()
#     audio.export(raw_audio, format="wav")
#     raw_audio.seek(0)
#     
#     # Read back as numpy array for sounddevice
#     data, sr = sf.read(raw_audio, dtype="float32", always_2d=True)
#     
#     # Determine output device and sample rate
#     output_device, device_sr = get_default_device("output")
#     
#     # Resample if needed
#     if sr != device_sr:
#         gcd = np.gcd(int(device_sr), int(sr))
#         up = device_sr // gcd
#         down = sr // gcd
#         data = resample_poly(data, up, down, axis=0)
#         sr = device_sr
#     
#     # Play audio
#     sd.play(data, samplerate=sr, device=output_device)
#     sd.wait()


# def play_audio(file_path):
#     try:
#         if SYSTEM == "Darwin":
#             os.system(f"afplay '{file_path}'")
#         elif SYSTEM == "Windows":
#             os.system(f'start "" "{file_path}"')
#         elif SYSTEM == "Linux":
#             os.system(f"aplay '{file_path}'")
#         else:
#             logger.error(f"Cannot play audio automatically on {system}. Please open {file_path} manually.")
#     except Exception as e:
#         logger.error(f"Audio playback failed: {e}")

def play_audio(filename, *, label='audio', triggered_at=None, capture_ready_at=None,
               speech_ended_at=None):
    preparation_started = time.perf_counter()
    # Load audio, apply high pass filter
    audio = AudioSegment.from_file(filename)
    audio = audio.high_pass_filter(HI_PASS_FREQ)

    # Export to raw data for playback
    raw_audio = io.BytesIO()
    audio.export(raw_audio, format="wav")
    raw_audio.seek(0)
    
    # Read back as numpy array for sounddevice
    data, sr = sf.read(raw_audio, dtype="float32", always_2d=True)

    # # Read file as float32, always 2D
    # data, sr = sf.read(filename, dtype="float32", always_2d=True)

    # Ensure stereo
    if data.shape[1] == 1:
        data = np.tile(data, (1, 2))

    # ALSA USB output
    output_device, device_sr = get_default_device("output")
    # output_device = "hw:3,0"

    # Resample if needed
    if sr != device_sr:
        gcd = np.gcd(int(device_sr), int(sr))
        up = device_sr // gcd
        down = sr // gcd
        data = resample_poly(data, up, down, axis=0)
        sr = device_sr

    # normalize audio
    peak = np.max(np.abs(data))
    if peak > 0:
        data = data / peak  # scale so max amplitude is 1.0

    # Play and wait
    stream_started = time.perf_counter()
    sd.play(data, samplerate=sr, device=output_device)
    playback_started = time.perf_counter()
    # This is the software stream start, not a measurement at the loudspeaker.
    logger.info('Timing: %s playback started; preparation %.3fs, stream startup %.3fs, audio duration %.3fs.',
                label, stream_started - preparation_started, playback_started - stream_started, len(data) / sr)
    if triggered_at is not None:
        logger.info('Timing: trigger to %s playback start %.3fs.', label, playback_started - triggered_at)
    if capture_ready_at is not None:
        logger.info('Timing: capture ready to %s playback start %.3fs.', label, playback_started - capture_ready_at)
    if speech_ended_at is not None:
        logger.info('Timing: estimated speech end to %s playback start %.3fs.',
                    label, playback_started - speech_ended_at)
    sd.wait()
    logger.info('Timing: %s playback finished; stream wait %.3fs.', label, time.perf_counter() - playback_started)

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
        device = choose_input_device(devices, sd.default.device[0])
        return device, int(devices[device]['default_samplerate'])
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
# Entry Point
# ------------------------------------------------------------
if __name__ == "__main__":
    run()
