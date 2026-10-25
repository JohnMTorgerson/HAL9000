# Installing HAL

Use Python 3.12 (the project's `.python-version` selects 3.12.11). The full
runtime entry point is `src/requirements.txt`; it includes the wake, image,
news, and live-transcription requirements. Installing a feature-specific file
alone is not a complete installation.

## Apple Silicon Mac

Install Apple's Command Line Tools if they are not already installed
(`xcode-select -p` checks; `xcode-select --install` installs), then:

```sh
brew install ffmpeg
```

FFmpeg is an external executable used for audio decoding, not a pip package.
The macOS sounddevice and soundfile wheels provide PortAudio and libsndfile;
separate Homebrew installs of those libraries are normally unnecessary.
Command Line Tools are needed for packages distributed as source, including
the existing simpleaudio diagnostic dependency.

From the repository root, create a new environment on each computer. Keeping
it in your home directory lets Macs share the source on an external SSD
without sharing interpreter paths or installed native libraries:

```sh
pyenv install -s 3.12.11
PYENV_VERSION=3.12.11 pyenv exec python -m venv "$HOME/.venvs/hal"
source "$HOME/.venvs/hal/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r src/requirements.txt
python -m pip check
```

Do not reuse an old environment copied from another computer. When using an
IDE, select `$HOME/.venvs/hal/bin/python` as its interpreter.

Allow microphone access for the terminal/IDE launching HAL. The spacebar
listener may also need Accessibility/Input Monitoring permission.

## Raspberry Pi 5 / 64-bit Raspberry Pi OS

Use a Python 3.12 environment and a 64-bit OS (`uname -m` should report
`aarch64`). This dependency set does not claim support for 32-bit Pi OS.
Native runtime libraries and build tools are separate from pip:

```sh
sudo apt update
sudo apt install ffmpeg libportaudio2 libsndfile1 build-essential python3-dev linux-libc-dev libasound2-dev swig liblgpio-dev
```

Then create/activate the environment and install `src/requirements.txt` as
above. The requirements select `evdev` and `gpiozero` on Linux and `lgpio` on
Linux ARM64; macOS skips them. Conversely, Linux skips the PyObjC packages.
If using pyenv, its Python build prerequisites must be installed first; use
headers matching the interpreter when compiling extensions.

For a physical LED, set `PLATFORM=pi` in the existing `.env` and use the Pi's
GPIO permissions. `GPIOZERO_PIN_FACTORY=lgpio` explicitly selects the Pi 5
backend. The keyboard listener needs access to the appropriate `/dev/input`
device. Missing hardware permissions are not fixed by reinstalling pip
packages. Other machines normally use HAL's dummy LED.

## Models, configuration, and launch

The requirements install spaCy's `en_core_web_sm` 3.8.0 model together with
spaCy 3.8.7 and preshed 3.0.10. No separate `spacy download` step is needed.
The model is a versioned, hash-checked wheel from the official spaCy models
repository.

Keep your existing `.env` private and ensure its file paths work on the new
machine. Preserve the Piper voice and JSON config, HAL audio clips, memory
data, display media, and log directories. A fresh pip installation cannot
recreate private API keys or personal data. iCloud may require a new sign-in
and two-factor code on a new machine.

Prepare/verify the Whisper wake models, then launch from `src`:

```sh
python src/setup_wake.py
cd src
python hal.py
```

Model preparation reuses suitable existing model files/downloads. Local
query transcription may download its configured Whisper model on first use.
Piper voice files are separate from the `piper-tts` Python package.

Only the Ollama backend needs the external `ollama` application/service and
the model selected in `.env`; the OpenAI backend does not need Ollama.

## Development and historical scripts

To install the runtime plus test tools, from the repository root:

```sh
python -m pip install -r src/requirements-dev.txt
python -m pytest tests
```

The browser tests are opt-in and need a separate Chromium download:

```sh
python -m playwright install chromium
HAL_BROWSER_TESTS=1 python -m pytest tests/test_display_connection_browser.py tests/test_image_display_browser.py
```

On Linux, Playwright may also require its documented system browser libraries
(`python -m playwright install-deps chromium`). Run tests from `tests/` as
shown: `src/whisper_test.py` is an interactive audio diagnostic, not a unit
test. Its `simpleaudio` dependency is retained in the runtime requirements.

`src/hal-press_space_to_record.py` is a historical Porcupine entry point, not
the current HAL application. It imports `pvporcupine` and uses old application
APIs; installing Porcupine alone does not make it current. The supported
`src/hal.py` uses Whisper wake detection and does not need a Picovoice account
or Porcupine package. Test-only pytest and Playwright are in the dev file.

## Dependency maintenance

Most existing pins are retained. The newly declared pyicloud version requires
`certifi>=2026.4.22` and `pydantic>=2.13.4`. Pydantic now selects its matching
`pydantic-core` and `typing-inspection` rather than conflicting with the old
independent pins. New direct dependencies include calendar/date parsing,
Wikipedia parsing, sports time zones, NLP/model loading, and the display
server. The live requirements also supply Uvicorn's WebSocket transport.
The spaCy/preshed pair is pinned after a real model-load check: preshed 3.0.14
caused `OverflowError` in the tokenizer with both spaCy 3.8.7 and 3.8.16;
preshed 3.0.10 with spaCy 3.8.7 loaded and processed text successfully.

This is a runtime requirements list with some retained version pins, not a
fully locked environment. After installation, `python -m pip check` verifies
installed package metadata; audio, GPIO, permissions, private services, and
model inference still need testing on the actual Mac or Pi.

Validation for this update: Python 3.12 dependency resolution succeeded for
Apple Silicon macOS and Linux ARM64 (including the dev requirements on Linux).
In a fresh Linux Python 3.12.14 test environment, installed dependencies passed
the package-consistency check; real NLP/model, calendar module, Wikipedia,
sports, display, and WebSocket imports passed; the test suite reported 444
passed, 10 skipped, and 147 subtests passed. The environment installed the
test/import dependency subset under the runtime constraints, not the entire
audio/inference stack. Browser opt-in tests, Mac audio/permissions, physical
Pi GPIO, and live account authentication were not exercised.
