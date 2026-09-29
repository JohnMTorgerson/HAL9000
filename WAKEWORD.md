# Local Whisper wake detection

HAL now uses local faster-whisper for “Hey HAL” detection. Porcupine is no
longer imported or initialized by `src/hal.py`; no Picovoice account is needed.
Query transcription still uses the existing `WhisperSTT` backend and model.
For the current setup, keep:

```dotenv
TRANSCRIPTION_BACKEND=local
WHISPER_MODEL_NAME=base
```

## Update an existing HAL installation

Use the same Python environment that already runs HAL. From the repository
root, on the Mac or Raspberry Pi:

```bash
git pull --ff-only
python -m pip install --only-binary=:all: -r src/requirements-wake.txt
python src/setup_wake.py
cd src
python hal.py
```

`setup_wake.py` reuses the pinned model in a sibling
`HAL9000-Whisper-Wake-Test/models` folder when present. Otherwise it downloads
base.en once (about 145 MB of weights). Completed downloads are reused on a
retry. It verifies model initialization before reporting success. The
`src/wake-models` cache is ignored by Git. Listening itself is offline.

The new compiled dependencies have Python 3.11/3.12 wheels for Apple Silicon
and 64-bit Linux ARM. Keep the Pi's existing working HAL Python environment;
use a 64-bit OS for the ARM wheels. This is an incremental dependency install,
not a request to recreate the full HAL environment. If PortAudio is missing
on Linux, install the distribution's `libportaudio2` package.

On the Mac, the default input is the system microphone, as before. On Linux,
HAL's existing USB-microphone selection is retained. The selected name and
sample rate appear in the log. To select the FIFINE explicitly, add:

```dotenv
HAL_INPUT_DEVICE="fifine Microphone"
HAL_INPUT_CHANNEL=1
```

A device number is also accepted. List devices with:

```bash
python -m sounddevice
```

Existing `PICOVOICE_ACCESS_KEY`, `KEYWORD_FILE_PATH`, and `SILENCE_THRESHOLD`
entries are no longer used by the main voice input path; they do not need to
be removed to run the update. `src/hal-press_space_to_record.py` is an older,
separate script and is not changed by this integration.

## Behavior

Default detection uses base.en on CPU with INT8, a three-second rolling
window, a 0.75-second check interval, and up to +24 dB analysis gain. Accepted
transcriptions are the whole-word pairs “hey hal”, “hey hall”, “hey hell”,
“hey how”, “hey al”, and “hey howl”. The phrase is matched within one segment.
The model receives no prompt or hotword suggesting the phrase.

A dedicated microphone reader continues collecting audio while inference
runs. If a scan is slow, the next scan uses fresh audio instead of building a
queue of old windows. The command includes the triggering window and one
extra second of history, then continues through the rest of the sentence.
There is no need to pause after “Hey HAL”. Speech detection ends the command
after approximately 1.2 seconds without speech. The full transcript, including
the wake phrase, is passed to HAL; no phrase-removal rule can remove a query
word such as “how”.

Holding spacebar still activates recording; releasing it ends the recording.
Press and release are watched together, including while wake inference is
busy. Spacebar takes priority when it is pressed during a wake scan. Both
paths use the same continuous capture and close the microphone before query
transcription and HAL's reply, retaining the existing turn-taking behavior.

The stream is allowed to finish its current short read before stopping. This
avoids the abort-before-join ordering that left a blocked reader in the
standalone Mac live test. Streams and keyboard listeners are closed on normal
completion, interruption, or error.

Commands have a 25-second limit, including buffered history, within the
existing local transcriber's 30-second audio limit. An overlong request is
discarded with a message so a partial instruction is not sent to the LLM.
Input overflow or loss of required audio is reported as an error rather than
silently transcribing a damaged command.

If the wake model or its dependencies are unavailable, startup logs the
problem and spacebar remains available. Install the wake dependencies and run
`setup_wake.py`, then restart HAL. You can deliberately select spacebar-only
mode with `WAKE_ENABLED=false`.

## Optional tuning

The tested defaults require no new `.env` entries. Available settings:

| Setting | Default | Purpose |
| --- | --- | --- |
| `WAKE_ENABLED` | `true` | Enable local wake detection |
| `WAKE_MODELS` | `base.en` | `base.en`, `tiny.en`, or both separated by spaces/commas |
| `WAKE_THREADS` | `2` | Wake model CPU threads |
| `WAKE_WINDOW_SECONDS` | `3` | Length of each rolling window |
| `WAKE_HOP_SECONDS` | `0.75` | Minimum interval between scan endpoints |
| `WAKE_SILENCE_SECONDS` | `1.2` | Silence before ending a detected request |
| `WAKE_MAX_GAIN_DB` | `24` | Gain cap for wake analysis |
| `WAKE_NORMALIZATION` | `capped` | `capped` or full `peak` normalization |
| `VOICE_MAX_SECONDS` | `25` | Maximum buffered command length |
| `WAKE_MODEL_DIR` | `src/wake-models` | Override the model cache |

For a tiny comparison, set `WAKE_MODELS=tiny.en`, run `python src/setup_wake.py`
from the repository root, then start HAL normally. `WAKE_MODELS="tiny.en base.en"` runs both models sequentially and accepts either; it costs extra
processing. Query transcription remains controlled by `TRANSCRIPTION_BACKEND`
and `WHISPER_MODEL_NAME`.

INFO logs show microphone selection, the wake model, per-detection processing
time, capture duration, and skipped scan slots. A scan's processing time is
not total delay from the spoken wake phrase. On a slower Pi, skipped windows
can still cause misses; Pi performance needs an actual device run.

## Validation

The standalone Mac live run produced 13 detections and 13 complete command
transcriptions with no microphone overflows. The user reported no missed
attempts or deliberate-background triggers. Median wake-decision processing
was about 0.61 seconds, and median existing query transcription about 0.59
seconds. This short session does not establish long-term reliability.

The integrated detector also matched all 13 saved trigger windows in Linux
replay. Ten hardware-free checks cover buffer wrapping and loss detection,
resampling, phrase boundaries, overflow reporting, graceful reader shutdown,
press/release capture, missing-model fallback, full utterance buffering, and
spacebar priority during slow inference. Run them with:

```bash
python -m unittest discover -s tests -p test_voice_input.py -v
```

The new shutdown order is verified with an instrumented blocking stream;
native Mac shutdown and full HAL operation on the Pi still need device testing.
The existing query transcription, LLM, iCloud, and TTS integrations are not
replaced. Live test audio/transcripts are not included in this repository.
