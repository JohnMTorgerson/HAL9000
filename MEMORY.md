# Persistent conversational memory

HAL can keep recent conversation across restarts and gradually learn a compact
set of personal facts and current topics. This feature is **off by default**.
It makes additional paid OpenAI calls in the background. It uses the installed
OpenAI package and Python's standard library; no database server, vector model,
or additional dependency is required on the Mac or Pi.

## Enable it

Add these settings to the same `.env` HAL already loads:

```dotenv
MEMORY_ENABLED=true
MEMORY_MODEL=gpt-6-luna
MEMORY_REASONING_EFFORT=medium
```

Restart HAL. Look for `Persistent memory ready:` and the storage directory.
The foreground still uses `LLM_MODEL`, its existing fast tier, and Luna's
`reasoning_effort=none`. Background memory uses its own client, ordinary
`service_tier=default`, and the configured reasoning effort. Higher reasoning
may improve decisions, but does not guarantee correctness and can cost more.
There is no extra memory-selection call before HAL answers.

Optional settings (defaults shown):

```dotenv
MEMORY_DIR=data/memory
MEMORY_MAX_COMPLETION_TOKENS=8192
MEMORY_TIMEOUT_SECONDS=120
MEMORY_SOFT_TOKEN_BUDGET=3000
```

`MEMORY_MODEL` defaults to `LLM_MODEL` when omitted, then `gpt-6-luna` if neither
is set. Use an OpenAI model supporting both reasoning effort and structured
JSON output. Memory currently requires `LLM_BACKEND=openai`.

`MEMORY_REASONING_EFFORT` accepts `none`, `low`, `medium`, `high`, `xhigh`, or
`max` for Luna. Start with `medium`; changing it does not change foreground
reasoning. The completion limit includes reasoning tokens **and** the JSON
result. If `memory.log` reports truncated decisions, increase the limit to
`16384`, or reduce effort. HAL never applies partial results or silently retries
an incomplete paid request. Actual API usage, including reasoning tokens when
reported, appears in `memory.log`.

## Files and manual editing

Relative `MEMORY_DIR` paths are relative to the HAL9000 repository, regardless
of the directory used to start HAL. The default Pi files are:

```text
/home/pi/Projects/HAL9000/data/memory/memory.json
/home/pi/Projects/HAL9000/data/memory/recent_conversation.json
```

`memory.json` contains `personal` and `topics` arrays. Each entry has a stable
`id`, short `text`, `basis` (`explicit` or `inferred`), dates, and a few supporting
user quotes. Personal entries have `expires_on: null`. Topics use an ISO date,
such as `2026-10-31`; without a more appropriate event date, the updater uses
30 days. A topic is included through its expiry date and excluded afterward.
Expired topics are removed on the next successful background update. Personal
facts do not expire for age or inactivity.

**Stop HAL before editing.** Edit entry text/basis as needed, or remove an entry
from its array, preserving valid JSON and the other fields. Leave `version`,
`last_processed_turn`, `context_after_turn`, and existing IDs intact. Then restart.
For a new entry, the simplest route is to tell HAL the fact, then edit its entry.
The two files are local data, excluded from Git; pulling a code update does not
overwrite them. The Mac and Pi have independent memory unless you copy it.
To move memory, stop HAL on both machines and copy **both JSON files together**.

Writes use atomic replacement and keep the previous version as `.json.bak`.
Malformed files are not reset to empty: HAL logs the problem and continues
without persistent memory. Correct the JSON, or restore a valid backup while
HAL is stopped. A second HAL instance cannot write the same memory directory.
Edits detected while HAL is running are rejected rather than overwritten.

The recent file usually contains the last `LLM_MAX_HISTORY` completed exchanges.
It can also retain older **pending** exchanges while an update is unfinished or
failed. Only the most recent configured number are supplied to the foreground.
Completed exchanges are timestamped, so yesterday's plans are not presented as
today's live information. There is no import of old logs or raw microphone audio.

## What is remembered

Only accepted speech and the final completed spoken response enter this store.
Ignored/end follow-ups, failed turns and intermediate external API payloads are
excluded. The assistant's reply helps interpret references but is never accepted
as evidence for a personal fact. All speech is assumed to be from **Torgo**;
there is no speaker identification in this version.

After playback finishes, a single background worker processes one new exchange
per API call, in order, with existing memories and a few earlier exchanges for
context. A request
can propose additions, updates, reinforcement, deletions, or no changes. Every
change must cite an exact quote from a new accepted user turn. Application code
validates the entire patch before writing anything; semantic judgments still
depend on the model and should be checked in the log.

The updater is instructed to preserve explicit facts, mark uncertain inferences,
merge duplicates, correct contradictions, and ignore routine weather/time facts.
Recurring interests should be supported across separate days; one trivia query
does not prove fandom. Reinforcement records evidence dates, and processing a
saved exchange again after a crash cannot count it twice. Corrections replace
old supporting evidence rather than treating it as support for the new claim.

You can say “Remember that my cat is named Miso,” “Actually, his name is Milo,”
or “Forget my cat's name.” Ordinary mentions can also become memories. A forget
decision deletes matching personal entries and clears active topic notes and
earlier recent context, to prevent them from recreating the detail. This happens
asynchronously; look for completion in `memory.log`. New statements after that
request can still be learned. Forgetting is not erasure of diagnostic logs or
backup files, which you can manage separately. Per-conversation privacy mode
is not implemented; use `MEMORY_ENABLED=false` and restart to disable this feature.

## Recall, responsiveness and failures

Before each new spoken request, HAL reads an in-memory snapshot containing all
personal memories and unexpired topics, plus recent conversation. These are sent
with the persona in the existing foreground request. Memory is explicitly labeled
as data, not instructions, and inferred information is identified as tentative.
An external-API continuation uses the same snapshot for consistency.

The background worker never holds a storage lock while waiting for the API.
HAL can continue answering and recording completed turns during a slow update.
There is a small local save after playback, but no foreground wait for reasoning.
New information remains in recent conversation while the background job catches up.

API failures leave existing memory and pending exchanges intact. The next
completed exchange or next startup triggers another attempt; there is no timed
retry loop. File/disk failures are logged and normal HAL conversation can continue.
Shutdown waits at most one second for the worker; pending work resumes later.
Updating memory and marking its input processed is one atomic write, so a crash
between that write and recent-history cleanup cannot apply a change twice.

`MEMORY_SOFT_TOKEN_BUDGET` is a warning threshold, estimated from character count,
not an exact tokenizer or a deletion limit. All active memories remain included
if it is exceeded. Individual entries are limited to 800 characters. No selective
retrieval is performed in this version; use actual memory size and timing results
to decide whether it is needed later.

## Separate memory log

Detailed records are written to **`memory.log` inside `LOG_PATH`**, alongside
`log.log` and `error.log`. It works with `DEBUG_ON=False`, contains no color codes,
and does not get pushed to HAL's display. It records:

- Accepted exchanges queued for processing and the IDs of each update batch.
- Add/update/reinforce/delete decisions with before/after entries, supporting
  quotes, and a short explanation. No-change batches are recorded too.
- Topic expiry, forget requests, rejected patches, failures and API token usage.

These are concise explanations of decisions, not the model's private reasoning.
The main HAL log/terminal gets a brief completion or failure message. The separate
file rotates at approximately 2 MB, keeping three previous files (`memory.log.1`
through `.3`). Inspect it with `tail -f /your/LOG_PATH/memory.log`, substituting
your existing log directory.

## Validation

Automated tests use temporary JSON stores and mocked HTTP responses with the
installed OpenAI SDK; they make no paid calls. They exercise restart persistence,
fact correction, reinforcement, expiry, forgetting, evidence validation, crash
recovery, concurrent recording during a stalled update, and foreground/follow-up
integration. Live model decision quality still needs evaluation with your speech.
