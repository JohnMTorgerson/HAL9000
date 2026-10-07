# Local Whisper wake detection

HAL now uses local faster-whisper for “Hey HAL” detection. Porcupine is no
longer imported or initialized by `src/hal.py`; no Picovoice account is needed.
Query transcription is configured separately. `TRANSCRIPTION_MODE=static` is
the default: HAL finishes recording before transcribing the complete file,
using your existing `TRANSCRIPTION_BACKEND=local` or `api` setting.
Optional live API transcription is described below; updating HAL does not enable it.

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

Automatic input selection prefers FIFINE, then USB microphones, then a
nonvirtual system default or another available input. Names associated with
virtual inputs (including Virtual Desktop Mic, BlackHole, Loopback and
aggregate devices) are skipped. This is a name-based heuristic: pin the
microphone explicitly for a predictable choice. The selected name and sample
rate appear in the log. To select the FIFINE, add to your existing `.env`:

```dotenv
HAL_INPUT_DEVICE="fifine Microphone"
HAL_INPUT_CHANNEL=1
```

A device number is also accepted, but names remain useful when device numbers
change after reconnecting hardware. An explicit setting always takes priority,
including when intentionally selecting a virtual input. A missing or ambiguous
explicit device reports an error rather than choosing another microphone.
If automatic selection finds only virtual inputs, it asks for an explicit
choice. List devices with:

```bash
python -m sounddevice
```

Existing `PICOVOICE_ACCESS_KEY`, `KEYWORD_FILE_PATH`, and `SILENCE_THRESHOLD`
entries are no longer used by the main voice input path; they do not need to
be removed to run the update. `src/hal-press_space_to_record.py` is an older,
separate script and is not changed by this integration.

## iCloud authentication

iCloud verification identifies the active delivery method. If it says SMS, use
the text-message code; device pop-up codes belong to a different verification
route. Apple/pyicloud may still notify multiple trusted devices or fall back to
SMS. HAL makes one initial code request and requests another only when you enter
`r`. It does not choose which individual Apple device receives a prompt.

A rejected code or verification exception allows another attempt, up to three
submissions. In pyicloud 2.6.5 a trusted-device attempt closes its verification
session, so HAL asks you to enter `r` for a fresh code before trying that route
again. Resends are also limited to three code requests per run.

Press Enter or enter `s` at the code prompt to skip iCloud Calendar and continue
using HAL. Exhausted attempts, unavailable input, and sign-in/network failures
also leave the calendar unavailable for that run without stopping HAL. Calendar
requests then report unavailability instead of an empty schedule. Restart HAL
to sign in again. The existing `accept_terms=True` setting is retained.

## OpenAI API errors

Wake detection and local query transcription do not use OpenAI API credits.
When `LLM_BACKEND=openai`, the language model still requires API access.
`credit_balance_exhausted` means credits must be added at
https://platform.openai.com/settings/organization/billing/. Retrying does not
restore an exhausted balance. `insufficient_quota` can also indicate an account
limit; temporary request rate limits have a different recovery message.

HAL reports service errors in the console and display, turns off the LED, and
returns to listening. Failed requests are not added to conversation history.
Each LLM request makes one API attempt, with no automatic retries. After fixing the
billing or connection problem, ask again; an API key change needs a restart.
No microphone or model reinstall is needed for a billing error.

## Choosing the OpenAI language model

To use GPT-6 Luna, set the following in the existing `.env` and restart HAL:

```dotenv
LLM_BACKEND=openai
LLM_MODEL=gpt-6-luna
```

HAL sends `reasoning_effort="none"` for this model to preserve quick replies,
the current 512-token output budget, and compatibility with `temperature=1`.
See the [Luna model page](https://developers.openai.com/api/docs/models/gpt-6-luna)
and [GPT-6 migration guidance](https://developers.openai.com/api/docs/guides/latest-model).
Use an account with access to the model and available API credits.

### Testing Luna Fast mode

To request Fast mode for HAL's LLM calls, add this to the existing `.env`
and restart HAL:

```dotenv
LLM_SERVICE_TIER=fast
```

The model stays `gpt-6-luna`, with reasoning disabled. The setting applies to
both the initial reply and follow-up replies after calendar/weather requests.
Query transcription has its own API client and is unaffected. For a comparison,
keep `TRANSCRIPTION_BACKEND=api`, `WAKE_MODELS=tiny.en`, and `WAKE_BEAM_SIZE=5`.

HAL sends the API's equivalent `priority` value, which is supported by the
existing pinned OpenAI SDK; no package upgrade is needed. Each completed LLM
request logs the requested and actual service tiers at INFO, independently of
`DEBUG_ON`. `used=fast` or `used=priority` confirms Fast processing;
`used=default` means standard processing, and `used=not reported` means the
response did not identify its tier. HAL does not silently retry on a different
tier if the API rejects the request.

Compare the `Timing: initial LLM response` and `Timing: follow-up LLM response`
lines across repeated similar queries. Faster LLM processing may reduce the
response delay, but wake detection, recording, transcription, and audio playback
still contribute to the total.

As of September 30, 2026, Luna Fast mode costs twice its standard token rates.
See the [Fast mode guide](https://developers.openai.com/api/docs/guides/fast-mode)
and [Luna pricing](https://developers.openai.com/api/docs/models/gpt-6-luna).
To explicitly return to standard processing, set `LLM_SERVICE_TIER=default`
and restart. Leaving the setting absent or blank preserves the API's project
default, as before; `auto` also uses the project setting. A project configured
for Fast mode may therefore still use Fast when the setting is absent.

### Listing available models

To list model IDs returned by the API for HAL's key, run this from `src` in the
same Python environment used for HAL. It searches for `.env` from the current
directory upward and prints model IDs without displaying the key:

```bash
python - <<'PY'
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

load_dotenv(find_dotenv(usecwd=True))
with OpenAI() as client:
    for model in sorted(client.models.list(), key=lambda model: model.id):
        print(model.id)
PY
```

The list includes models for other tasks, such as audio and embeddings; model
presence alone does not establish Chat Completions compatibility or pricing.

## Optional conversational follow-ups

After an explicit wake or spacebar request, HAL can listen briefly for another
utterance without requiring the wake phrase again. This feature is off by
default. Pull the update in your existing HAL environment; no additional
dependencies or model downloads are needed for an installation with working
Whisper wake detection.

To enable it, add these entries to your existing `.env` and restart HAL:

```dotenv
FOLLOWUP_ENABLED=true
FOLLOWUP_WINDOW_SECONDS=30
```

Keep `TRANSCRIPTION_MODE=static` and your working transcription, wake, and Luna
settings. Follow-up listening uses local speech detection, without running the
Whisper wake decoder for each utterance in the open window. Captured speech
goes through the existing query transcriber. The feature requires
`LLM_BACKEND=openai` and a model supporting structured outputs, such as
`gpt-6-luna`. It preserves the configured Fast tier and Luna's disabled reasoning.

The window opens after HAL's **final reply finishes playing**, never during his
voice or after an intermediate “Just a moment.” It lasts thirty seconds by
default, including microphone startup. Speech that starts before the deadline
can finish afterward, subject to the existing command-length limit. A short
buffer-delivery allowance accommodates microphone latency at the boundary;
it does not admit speech starting after the deadline.

Luna receives the candidate transcript and accepted conversation history in one
combined decision/response request. Its structured result has three outcomes:

| Decision | Behavior |
| --- | --- |
| `respond` | Use the normal reply or external-request path; open a new short window after the final spoken reply. |
| `ignore` | Stay silent, leave the original deadline unchanged, and listen again for any time still remaining. |
| `end` | Stay silent and close the window immediately; require “Hey HAL” or spacebar again. |

Capturing, transcription, and classification consume time from the existing
window; ignoring a remark never pauses or renews that timer. Rejected/end
utterances are excluded from conversation history and the display transcript,
and cannot trigger TTS, Wikipedia fallback, or calendar/weather requests.
Malformed, truncated, refused, or failed model decisions close the session and
report an error without treating the raw output as a reply. No second LLM call
is made to classify an otherwise accepted follow-up; normal external requests
still use their existing follow-through call.

There is no overall session time limit. Every accepted exchange renews the full
configured window after HAL finishes his reply, including an utterance that
started just before the old deadline and finished afterward. A conversation can
continue for as long as accepted exchanges continue. Silence or ignored speech
does not renew the window; on expiry, a fresh explicit request is needed.
During an open window, a clear “Hey HAL” in the query transcript tells Luna to
treat it as directly addressed. The ambiguous “hey how”
variant still works in idle wake detection but does not override the follow-up
filter by itself. Spacebar retains priority and uses the ordinary unfiltered
request path. End phrases such as “That's all, HAL” or “Stop listening” can be
spoken as automatic follow-ups.

Luna is instructed to accept clear continuations, corrections, answers to its
questions, and assistant-directed changes of subject; uncertain background
dialogue is ignored. Short answers such as “Not really,” “Probably,” or “I'm not
sure” are interpreted in the context of HAL's last question; a negative answer
alone does not end the conversation. Declining an offered lookup should not
perform that lookup or immediately repeat the offer. There is no separate timer
or extra classification call for answers to HAL's questions.
It cannot reliably distinguish identical words spoken by
you versus a TV character. Test this with your actual room and conversations.
Ignored speech cannot keep the window open. Speech mistakenly accepted as
addressed to HAL can renew it, so the local microphone setting and filter quality
still matter. “Stop listening” closes it explicitly.

| Setting | Default | Accepted values |
| --- | --- | --- |
| `FOLLOWUP_ENABLED` | `false` | `true` or `false` |
| `FOLLOWUP_WINDOW_SECONDS` | `30` | 1–180 seconds |

Existing `.env` values override these defaults. `180` seconds is the maximum
accepted window override, not a session cap. The old `FOLLOWUP_SESSION_SECONDS`
setting is no longer used and can be removed; leaving it present has no effect.
At timeout HAL silently returns to wake/spacebar listening; there is no reminder,
extra question or model call to keep the conversation alive.

### Conversational style

HAL answers the request first and may occasionally add a relevant observation,
reasoned opinion, or one thoughtful follow-up question. The prompt uses context
and recent engagement rather than a probability, fixed frequency, or quota.
Concise factual replies remain appropriate. HAL should respond to the user's
answer before considering another question, avoid repetitive offers and avoid
turning every exchange into an interview.

Relevant supplied memories can connect a reply to previous discussions, the
user's interests, or HAL's own views. He may disagree politely or revise a view
with reasons; he must not invent shared experiences. Speculation is distinguished
from known facts. Ordinary disclosures and dilemmas invite conversation: saying
“I'm debating whether to go to choir practice tonight” must not trigger a calendar
lookup just because it mentions a plan. Actual schedule/time/conflict questions
still use the calendar protocol.

These instructions apply within existing response requests and preserve command-only
output for API/image/song actions. Natural observations may follow the requested
information in the final spoken reply. No autonomous conversation initiation,
new utterance-completeness judgment, or change to silence/spacebar capture is added.
The 30-second window is time to START speaking, not a new maximum utterance length.
The prompt's judgment and tone need evaluation in actual conversation; local tests
verify timers and request/control handling without paid model calls.

If background noise triggers a follow-up capture but transcription returns no
speech, HAL logs `FOLLOWUP heard: [empty transcription]` and resumes listening
for the remainder of the original window. Recording/transcription time counts
toward that deadline; an empty result neither closes it early nor restarts it.
No LLM call, spoken response, or memory update is made for an empty result.
Live transcription does not retry an empty result through the paid static
fallback. Real transcription/service failures still close the window.

Silence is processed locally and makes no transcription or LLM request. Speech
that is ultimately ignored can still incur API transcription and classification
charges. INFO logs identify the open window, `respond`/`ignore`/`end` decisions,
their timing, and window expiry. `FOLLOWUP heard:` records every candidate's
full transcript before classification, including speech later ignored or ending
the conversation, and candidates whose classification fails. These lines appear
in the terminal and `log.log` even with `DEBUG_ON=False`; they do not appear on
HAL's display unless accepted as a normal user request.

The transcript pane stays visible while an accepted response is being prepared
and spoken, including image searches and songs. Its 30-second inactivity timer
starts when HAL finishes. Empty, ignored, and end-of-conversation follow-ups do
not refresh it. Closely spaced display lines get a deferred push, normally
within 250 ms, including the last line of a burst; they never depend on a later
log message.
Failed display pushes retry after five seconds while the text is still current.
Delivery failures and recovery appear in the main log, without entering the
transcript itself. Long transcripts scroll to their newest lines when opened.

The pane includes explicitly designated conversation/display messages plus
warnings and errors. Conversation stays cyan, warnings are yellow, and errors
(including critical errors) are red. Severity travels with each message, so a
warning does not recolor earlier conversation. INFO and DEBUG records stay off
the display; file logs remain plain text.

The embedded display server keeps Uvicorn's startup/shutdown, HTTP request,
warning and error messages in the terminal. Its terminal handlers are installed
without reconfiguring or closing HAL's transcript/file handlers. Successful
GET polls of `/api/state` and `/api/images/status/…` stay silent; failed polls and
other requests remain visible.

In an interactive terminal, user transcripts (including wake and follow-up
transcripts) are cyan, and HAL's spoken replies are green. Saved logs and HAL's
display remain plain text. Color is disabled for redirected output, `TERM=dumb`,
or a nonempty `NO_COLOR` environment variable (for example,
`NO_COLOR=1 python hal.py`). No debug setting is required for color.

Debug playback is skipped for automatic
follow-up candidates, so it cannot repeat background speech aloud. Existing
`DEBUG_ON`/`DEBUG_PLAYBACK` recording settings can still save that candidate to
`last_command.wav`. To disable the feature, set `FOLLOWUP_ENABLED=false` and
restart; normal wake and spacebar requests continue to work.

## Optional persistent memory

For persistent recent conversation, personal facts, and topic notes, see
[MEMORY.md](MEMORY.md). Memory is independently opt-in, uses editable JSON, and
runs its reasoning updates after replies with decisions in a separate `memory.log`.

## Static API transcription language

With `TRANSCRIPTION_BACKEND=api` and `TRANSCRIPTION_MODE=static`, HAL sends an
English language hint by default. This also applies to an explicitly enabled
static fallback from live mode. With `en`, a short transcription prompt also
requests verbatim English without translation, and an empty result for no
intelligible speech. This is a transcription hint, not a guaranteed language
filter; it adds no extra API request. Startup logs record the effective hint
and whether that prompt is enabled. Configure it in `.env`, then restart HAL:

```dotenv
TRANSCRIPTION_LANGUAGE=en
```

Use a different ISO-639-1 code, such as `fr`, for another language. Set
`TRANSCRIPTION_LANGUAGE=` to omit the hint and allow automatic detection.
This setting only affects static API requests; live mode continues to use
`LIVE_TRANSCRIPTION_LANGUAGES`, and local Whisper keeps its existing behavior.
OpenAI documents language hints as improving accuracy and latency; measure the
actual improvement using HAL's transcription and total response timing logs.

## Optional live query transcription

Live mode sends query audio to OpenAI while the command is still being recorded.
It connects only after a wake detection or spacebar press, first sending the
buffered beginning of the command, then new audio as it arrives. Idle listening
uses the local wake model and sends no audio to the transcription API.
Only the final transcript reaches the LLM. Wake detection, Luna, and Piper keep
their existing roles; this changes query transcription only.

Install the optional transport in HAL's existing Python environment, from the
repository root:

```bash
git pull --ff-only
python -m pip install -r src/requirements-live.txt
```

Add or update these entries in your existing `.env`, then restart HAL:

```dotenv
TRANSCRIPTION_BACKEND=api
TRANSCRIPTION_MODE=live
LIVE_TRANSCRIPTION_FALLBACK=false
```

For a comparison, keep the other working settings unchanged, including
`WAKE_MODELS=tiny.en`, `WAKE_BEAM_SIZE=5`, `WAKE_SILENCE_SECONDS=0.8`, and
`LLM_SERVICE_TIER=fast`. To return to the previous upload workflow, set
`TRANSCRIPTION_MODE=static` and restart. Local transcription also requires
static mode; HAL rejects a live/local combination instead of silently enabling
paid API use. `WHISPER_MODEL_NAME` continues to apply only to local transcription.

| Setting | Default | Purpose |
| --- | --- | --- |
| `TRANSCRIPTION_MODE` | `static` | Complete-recording upload/local transcription, or `live` API audio streaming |
| `LIVE_TRANSCRIPTION_MODEL` | `gpt-live-transcribe` | Streaming transcription model; requires API project access |
| `LIVE_TRANSCRIPTION_DELAY` | `low` | Model latency setting: `minimal`, `low`, `medium`, `high`, or `xhigh` |
| `LIVE_TRANSCRIPTION_LANGUAGES` | `en` | Comma-separated language hints; blank omits the hint |
| `LIVE_TRANSCRIPTION_TIMEOUT_SECONDS` | `8` | Maximum wait after recording ends for the final result, 1–60 seconds |
| `LIVE_TRANSCRIPTION_FALLBACK` | `false` | Explicitly allow one complete-file API upload after a live failure |

Both static API transcription and live transcription are billed; local static
transcription does not use API credits. As of September 30, 2026,
[GPT-Live-Transcribe](https://developers.openai.com/api/docs/models/gpt-live-transcribe)
costs $0.017 per audio minute, approximately $0.0017 for six seconds. Buffered
history and silence are part of the audio sent. A cancelled or failed turn may
still incur charges for audio already sent. The logged committed duration is
an audio measurement, not a billing receipt.

With fallback disabled, a live failure reports the problem and resumes
listening without another transcription request. Enabling fallback may incur
both a live charge and a static upload charge; logs identify the fallback.
HAL does not reconnect/retry a live turn automatically, and static uploads
also use one attempt. Partial or damaged commands never reach the LLM.

Audio resampling and network work run in a background thread. Stateful
resampling converts the microphone's native PCM to 24 kHz without duplicating
or dropping samples between chunks. Live audio retains the microphone's input
level; static uploads retain the existing whole-recording normalization.
The existing local speech endpoint commits the live turn. It sends no more
audio after the command ends and closes the connection after the final result.
See the [Realtime transcription guide](https://developers.openai.com/api/docs/guides/realtime-transcription)
for the model's streaming behavior. No OpenAI SDK upgrade is required.

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
paths use the same continuous capture and close the microphone before HAL
processes the final transcript and replies. Static transcription starts after
capture; live transcription overlaps capture.

The stream is allowed to finish its current short read before stopping. This
avoids the abort-before-join ordering that left a blocked reader in the
standalone Mac live test. Streams and keyboard listeners are closed on normal
completion, interruption, or error.

Commands have a 25-second limit, including buffered history, within the
existing local transcriber's 30-second audio limit. An overlong request is
discarded with a message so a partial instruction is not sent to the LLM.
On microphone input overflow, HAL discards the entire capture, turns off the
LED, reports that the request needs repeating, and reopens the microphone after
a short pause. The display server stays running. Static mode never uploads a
discarded capture; live mode cancels the unfinished turn, whose earlier audio
may already have been sent. Neither mode forwards a damaged command to the LLM.
Other unexpected capture failures still surface as errors.

The input stream requests 0.25 seconds of buffering headroom instead of the
device's low-latency setting. This gives the reader more tolerance for scheduling
delays under CPU load; the reader still drains audio in short blocks. The actual
latency selected by the audio driver is logged when listening starts. If the Pi
continues to overflow, try `HAL_INPUT_LATENCY_SECONDS=0.5` in `.env` and restart.
More buffering can add input latency and does not guarantee that an overloaded
device will keep up.

If the wake model or its dependencies are unavailable, startup logs the
problem and spacebar remains available. Install the wake dependencies and run
`setup_wake.py`, then restart HAL. You can deliberately select spacebar-only
mode with `WAKE_ENABLED=false`.

## Optional tuning

Query playback is off by default, including when `DEBUG_ON=True`. To hear each
recorded query before transcription, set `DEBUG_PLAYBACK=True` in `.env` and
restart HAL. Leave it unset or set `DEBUG_PLAYBACK=False` for normal use.
`DEBUG_ON=True` still saves `last_command.wav` for inspection; enabling
`DEBUG_PLAYBACK` also saves that file even if general debugging is off.

No new `.env` entries are required. Wake decoding now defaults to beam size 2
to reduce search work. It retained all 13 saved wake detections in Linux replay;
beam size 1 retained only 11. Keep the same model and phrases while testing this
setting; set `WAKE_BEAM_SIZE=5` and restart to compare with the previous decoding
behavior. `WAKE_BEAM_SIZE=1` is available for experiments, with that observed
accuracy tradeoff. Speed and detection accuracy still need a live Pi comparison.

Available settings:

| Setting | Default | Purpose |
| --- | --- | --- |
| `WAKE_ENABLED` | `true` | Enable local wake detection |
| `WAKE_MODELS` | `base.en` | `base.en`, `tiny.en`, or both separated by spaces/commas |
| `WAKE_THREADS` | `2` | Wake model CPU threads |
| `WAKE_BEAM_SIZE` | `2` | Wake decoding search width, integer 1–10; previous setting was 5 |
| `HAL_INPUT_LATENCY_SECONDS` | `0.25` | Requested input buffering headroom, 0.02–2 seconds |
| `WAKE_WINDOW_SECONDS` | `3` | Length of each rolling window |
| `WAKE_HOP_SECONDS` | `0.75` | Minimum interval between scan endpoints |
| `WAKE_SILENCE_SECONDS` | `1.2` | Silence before ending a detected request |
| `WAKE_MAX_GAIN_DB` | `24` | Gain cap for wake analysis |
| `WAKE_NORMALIZATION` | `capped` | `capped` or full `peak` normalization |
| `VOICE_MAX_SECONDS` | `25` | Maximum buffered command length |
| `WAKE_MODEL_DIR` | `src/wake-models` | Override the model cache |
| `DEBUG_PLAYBACK` | `false` | Play the recorded query before transcription |

For a tiny comparison, set `WAKE_MODELS=tiny.en`, run `python src/setup_wake.py`
from the repository root, then start HAL normally. `WAKE_MODELS="tiny.en base.en"` runs both models sequentially and accepts either; it costs extra
processing. Query transcription remains controlled by `TRANSCRIPTION_BACKEND`
and `WHISPER_MODEL_NAME`.

With `TRANSCRIPTION_BACKEND=api` and static mode, queries use `gpt-4o-mini-transcribe`;
`WHISPER_MODEL_NAME` only selects the model when the backend is `local`.
Startup logs identify both the local wake model/beam size and the actual query
transcription backend/model, so the two paths can be distinguished in timing runs.

INFO logs show microphone selection, the wake model, per-detection processing
time, capture duration, and skipped scan slots. A scan's processing time is
not total delay from the spoken wake phrase. On a slower Pi, skipped windows
can still cause misses; Pi performance needs an actual device run.

Lines beginning `Timing:` are logged at INFO, including with `DEBUG_ON=False`.
They appear in the console and the normal log file, without going to the display
panel. They measure query audio preparation, transcription (including upload and
network wait in API mode), each LLM request, each external request, voice
synthesis, reply normalization, and playback preparation/start/finish. The
acknowledgment clip, final reply, and optional debug query playback have separate
labels. All durations use the monotonic performance clock.

Trigger-to-playback time begins at the LED trigger callback; capture-ready time
begins after the microphone has closed. These totals exclude the earlier wake
scan and waiting for the user. Playback start is measured just after the audio
output stream is started, not when sound physically reaches the speaker. The
finish line includes the time spent playing the reply, which is not response
latency. DEBUG_ON can remain off for timing tests; turn it on to save the latest
query as `last_command.wav`. Keep `DEBUG_PLAYBACK=False` when measuring normal
response time.

Live mode also logs connection setup time, the first partial transcript's timing
(without its text), and the final result's delay after recording ends. The
`transcription wait after capture` line measures only the remaining wait, since
earlier work overlaps recording. Check `Transcription source` to distinguish a
live result from a configured static fallback.

`Timing: TOTAL response latency` reports the delay from estimated speech end
until the first HAL audio playback starts, once per answered request. For an
external request it stops at the first “Just a moment”; for a direct answer it
stops at the reply. It includes silence/end-of-speech waiting, microphone cleanup,
transcription, LLM processing and any synthesis/playback preparation before that
first audio. Idle listening and reply duration are excluded. Ignored/end
follow-ups and debug query playback do not emit this total. If no speech endpoint
is available, the total explicitly uses `capture ready` instead and excludes
endpoint waiting. No new setting is required; it is logged at INFO.

`Timing: estimated speech end to ... playback start` retains the detailed totals. Use `acknowledgment` for an
external request and `reply` for a direct answer. The speech-end estimate uses
local VAD, the microphone read clock, and the driver's reported input latency;
it is not a physical speaker/microphone measurement. It is omitted if no speech
endpoint can be estimated. These new timing lines are also INFO, so `DEBUG_ON`
can remain off.

## Validation

The earlier beam-5 standalone Mac live run produced 13 detections and 13 complete command
transcriptions with no microphone overflows. The user reported no missed
attempts or deliberate-background triggers. Median wake-decision processing
was about 0.61 seconds, and median existing query transcription about 0.59
seconds. This short session does not establish long-term reliability.

On the same 13 saved trigger windows in Linux replay, beam sizes 5 and 2 both
matched 13/13; beam size 1 matched 11/13. Median scan time was 0.648 seconds for
beam 5 and 0.584 seconds for beam 2 in this comparison (about 10% faster). This
small replay covers previously detected windows; it does not establish live
recall or false-activation rates, and its timing is not a Pi benchmark.

Hardware-free checks cover buffer wrapping and loss detection,
resampling, phrase boundaries, overflow discard/recovery, graceful reader shutdown,
press/release capture, missing-model fallback, full utterance buffering, and
spacebar priority during slow inference. Run them with:

```bash
python -m unittest discover -s tests -p test_voice_input.py -v
python -m unittest discover -s tests -p test_hal_recovery.py -v
python -m unittest discover -s tests -p test_live_transcription.py -v
python -m unittest discover -s tests -p test_followup.py -v
python -m unittest discover -s tests -p test_icloud_auth.py -v
```

Additional tests cover physical-microphone preference, explicit input overrides,
credit exhaustion versus temporary rate limits, history preservation, and the
actual HAL loop returning to listening after a service failure. Decoding checks
cover beam sizes 1, 2, and 5 during warmup and live analysis. Simulated timings
verify that stage measurements exclude idle listening, reset for each request,
and record playback start before waiting for playback to finish. API tests use
the existing OpenAI SDK with a mock HTTP transport; they need no API key or credits.
The live suite additionally uses a local WebSocket server with the real transport
package. It checks streaming before the endpoint, exact resampling across chunk
boundaries, committed-item matching, timeout/cancellation, optional fallback,
and recovery after failed turns. Install `src/requirements-live.txt` to run it.

Follow-up tests cover silence expiry, speech starting just before/after the
deadline, spacebar priority, bounded session renewal, rejected history, strict
decision parsing, external-request handling, and returning to wake listening
after an end or failure. They use simulated microphone input and the installed
OpenAI SDK with mock responses; they do not measure Luna's real classification
accuracy or the Pi microphone's speech-onset reliability.

The shutdown order is verified with an instrumented blocking stream. The user's
Pi runs completed requests without input overflows after increasing the input
buffer. Subsequent tiny.en/API/Fast runs established a working static baseline.
Live transcription still needs a real API and Pi microphone comparison; the
automated checks do not establish account access, recognition accuracy, or a
latency improvement. Live test audio/transcripts are not included in this repository.
