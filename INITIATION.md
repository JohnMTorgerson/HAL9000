# HAL starting a conversation

HAL can occasionally ask whether you have a moment, then choose a conversational
opening using his memories and your recent conversations. This is opt-in and
requires working persistent memory, enabled followups, and `LLM_BACKEND=openai`.
Other backends leave ordinary conversation available and log that initiation is
disabled. There are no new dependencies.

## Enable and configure

Add this to the `.env` HAL already loads, then restart HAL yourself:

```dotenv
INITIATION_ENABLED=true
```

Keep your existing `MEMORY_ENABLED=true` and `FOLLOWUP_ENABLED=true` settings.
Optional settings, with their defaults:

```dotenv
INITIATION_START_HOUR=9
INITIATION_END_HOUR=21
INITIATION_DAILY_ATTEMPTS=1
INITIATION_QUIET_SECONDS=120
INITIATION_ACTIVITY_SECONDS=300
INITIATION_ACTIVITY_DB=-45
```

Hours use the Pi's local system timezone. The start is inclusive and the end is
exclusive; use integer hours with `0 <= start < end <= 24`. Overnight windows
are not supported. The daily maximum accepts 1 or 2 attempts. With two enabled,
the second opportunity is scheduled at least three hours after the first, if
enough time remains in the allowed hours. Unanswered invitations and declines
count as attempts. Missed days do not create a backlog.

Quiet and activity windows accept 30–3600 seconds. The activity threshold is
raw microphone RMS level in dBFS, accepting -80 to -10; a more negative value
notices quieter sounds. This is a simple cue that somebody might be nearby,
not reliable occupancy detection: a fan can qualify, and detected TV speech
can postpone an invitation. Start with the defaults and evaluate it in the room.

Set `INITIATION_ENABLED=false` to disable initiation. It is also disabled when
the variable is absent. Invalid configuration or storage errors are logged
without stopping ordinary HAL conversation.

## Program flow and API calls

1. **Choose an opportunity locally.** Pick a random time within the remaining
   allowed hours and save it. Once it is due, wait until HAL is idle, local speech
   detection has found no speech for two minutes, and there has been possible
   activity within five minutes. The activity cue is nonspeech microphone sound
   over the threshold, or a recent interaction with HAL. Wake/spacebar commands
   take priority. No extra STT or LLM request is made by this monitoring.
2. **Ask availability.** Save the attempt before speaking, close the microphone,
   and say “Torgo, do you have a moment?” using `HAL_USER_NAME` and HAL's existing
   local voice. This fixed sentence requires no LLM request. Show it as HAL speech
   in the transcript pane, then open the ordinary followup listening window.
3. **Interpret the answer and choose an opening in one request.** Transcribe
   captured speech normally. Empty results make no LLM request and keep the LED
   off. For nonempty text, one foreground call receives the fixed availability
   question, the actual reply, all compact memories, all retained recent
   transcripts, and recent initiation attempts. That call chooses one outcome:
   - Agreement: ask a short question or offer an observation.
   - Decline: acknowledge briefly and close the window.
   - A different request: answer it through HAL's normal workflow.
   - Unrelated or unclear background speech: ignore it without renewing the window.
4. **Continue normally.** After the opening finishes playing, save the actual
   exchange and renew the ordinary followup window. Subsequent replies use the
   existing relevance/response request and local memory retrieval. Memories used
   for the opening stay available throughout that followup conversation, even
   when a short answer would not match their keywords. Removed memories are not
   restored by this mechanism.
5. **Remember the attempt.** Save its status, topic, spoken opening, and referenced
   memory IDs locally. Record whether the user engaged or left the opening
   unanswered. Recent attempts help the next opening avoid repetition. Restarting
   does not reset the daily quota or resume an old pending invitation.

The listening window keeps its existing setting: 30 seconds by default,
`FOLLOWUP_WINDOW_SECONDS` overrides from 1 to 180 seconds, and no overall session
cap. Silence, empty transcription, and ignored speech do not extend it. Each
accepted reply starts a new window after playback.

A typical accepted invitation uses **one foreground LLM request** to interpret
“yes” and produce the opening. There is no selection or summarization request.
Silence uses none. Each nonempty background candidate can still require the
normal interpretation call. Accepted exchanges continue through the existing
background memory updater; those calls are separate. A new user request that
needs an external action can require its usual continuation calls.

## Memory supplied to the opening

Every `personal`, `topics`, and `hal` entry is included using its existing compact
summary, ID, tags, basis, and dates. Evidence quotes and maintenance metadata are
omitted. No new summary is generated. All retained recent transcripts are included,
including pending exchanges beyond the ordinary history length. The ordinary
memory selection budget does not filter this initiation catalogue.

The prompt encourages curiosity about known interests and things HAL has not yet
learned about you: experiences, motivations, tastes, outlook, or an unresolved
discussion. It asks HAL to check for existing answers, avoid interrogation and
repetition, treat inferred interests cautiously, and contribute remembered views
when appropriate. It does not let him invent private rumination or human biography.
An opening cannot run an unsolicited image, external API, or song command.

When an availability reply is accepted, its recent-conversation entry includes
an optional `assistant_lead_in` containing the actual preceding availability
question and timestamp. This preserves the meaning of “yes” without fabricating
a user utterance. Background memory validation accepts that lead-in only as
assistant evidence. Questions and unanswered invitations do not establish user
preferences or HAL beliefs.

## Try it without waiting

With HAL already running and initiation enabled, use another terminal on the Pi,
with the same Python environment you normally use for HAL:

```bash
cd ~/Projects/HAL9000
python src/initiate_conversation.py
```

This queues one test and **does not start HAL**. It uses the normal availability
question and response flow, bypassing scheduled hours, the activity requirement,
the long quiet period, and the daily maximum. It still waits until HAL is idle
and has detected no speech for at least one second. The request expires after
two minutes; if HAL is busy, it may expire before being used. Tests count toward
the day's ordinary attempts. `--memory-dir` can explicitly select the directory
if HAL was launched with a `MEMORY_DIR` environment override not in `.env`.

## Files and diagnostics

`initiation.json` lives beside `memory.json`, under `MEMORY_DIR`. It stores the
current schedule and the most recent 60 attempts; the last 10 attempts are used
for repetition context. It uses the existing storage lock, atomic replacement,
backup, and protection against overwriting edits made while HAL is running.
Stop HAL before editing it. Copy it too when moving memory to another machine
if you want to preserve initiation timing and history.

`initiate.request` is the short-lived local manual-test file. These files and
their backups are excluded from Git, including in custom memory directories.

The main log records startup configuration, invitations, window outcomes, and
failures. `memory.log` records the chosen schedule, the complete compact catalogue
supplied, included transcript IDs, the result and short selection explanation,
referenced memory IDs, and the saved opening. Followup retrieval logs identify
extra memories retained from that opening and the resulting supplied context size.

Existing forget operations also prevent older initiation text from being supplied
again as context. Attempt timestamps still count toward quotas. As with the
existing memory files, old text can remain in local backups and logs.
