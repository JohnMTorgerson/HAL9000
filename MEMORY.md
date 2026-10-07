# Persistent conversational memory

HAL can keep recent conversation across restarts, learn personal facts, remember
his own expressed views and interests, and preserve meaningful discussions.
This feature is **off by default**. It makes additional paid OpenAI calls in the
background. Recall searches local records; it never makes an LLM search request.
It uses the installed
OpenAI package and Python's standard library; no database server, vector model,
or additional dependency is required on the Mac or Pi.

## Enable it

Add these settings to the same `.env` HAL already loads:

```dotenv
MEMORY_ENABLED=true
MEMORY_MODEL=gpt-6-luna
MEMORY_REASONING_EFFORT=medium
```

Set `HAL_USER_NAME` in that same `.env` to your preferred name. The persona,
response cleanup, and both memory prompts use this setting, preserving spelling
and capitalization. If it is missing or blank, HAL uses `Dave`. Restart HAL after
changing it. Changing this setting does not rename facts already saved in JSON
or switch to a different user's memory; use a separate `MEMORY_DIR` for another user.

Restart HAL. Look for `Persistent memory ready:` and the storage directory.
The foreground still uses `LLM_MODEL`, its existing fast tier, and Luna's
`reasoning_effort=none`. Background memory uses its own client, ordinary
`service_tier=default`, and the configured reasoning effort. Higher reasoning
may improve decisions, but does not guarantee correctness and can cost more.
There is no extra memory-selection call before HAL answers. This update does
not change microphone capture, listening windows, or HAL's conversational
behavior; those remain separate features.

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

`memory.json` contains three sections:

| Section | Contents | Retention |
| --- | --- | --- |
| `personal` | Facts, preferences, interests, and views expressed by the user | Permanent until corrected or forgotten |
| `hal` | HAL's own expressed views and interests, with brief reasons where useful | Permanent, but revisable |
| `topics` | Dated shared conversations, plans, projects, and accumulating evidence of interests | Permanent until corrected, merged, or forgotten |

Each entry has a stable `id`, compact `text`, `basis` (`explicit` or `inferred`),
search `tags`, dates, and supporting evidence attributed to the user or HAL.
All three sections use `retention: durable` and `expires_on: null`. A plan ending
does not erase the conversation about it. Summaries include absolute discussion
and event dates, with unknown outcomes left unknown. Past plans, scores and injuries
must not be presented as current information. Records are not deleted because they
are old or were not selected by a search.

Existing version-1 and version-2 memory files are migrated locally: surviving
temporary topics, including already-expired ones still on disk, become durable.
IDs, text, tags, creation/update times, evidence and processing cursors are preserved;
only retention/expiry change. The prior file is backed up, and topic promotions
are logged. Already-deleted topics cannot be recovered by this migration; a
separate, explicit repair needs a transcript or backup. No LLM is needed for the
retention migration, and the file format remains version 2. Existing untagged entries
are queued for background tag enrichment, at most 20 per batch. This uses the
same updater and may make paid calls at startup even when there is no new spoken
exchange. Records remain available for text matching while tagging is pending.
Migration does not require feeding the full archive into every foreground prompt.

**Stop HAL before editing.** Edit entry text/basis/tags as needed, or remove an entry
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
excluded. HAL's reply may support a HAL-view entry or his side of a discussion
summary; it is never accepted as evidence for a personal fact about the user.
All speech is assumed to be from the user
named by `HAL_USER_NAME` in `.env`;
there is no speaker identification in this version.

After playback finishes, a single background worker processes one new exchange
per API call, in order, with locally selected existing memories and a few earlier
exchanges for context. The same call handles personal facts, HAL views, discussion
notes, and search tags; these do not each require a separate call. Pending tag
enrichment can accompany that exchange or run by itself at startup.

A request can propose additions, cumulative topic extensions, corrections,
reinforcement, deletions, or no changes.
Content changes must cite at least one exact quote from the newly accepted exchange
with the correct speaker role. They may also cite the supplied earlier dialogue
or still-valid evidence of any locally selected memory, including user quotes
from a topic when creating an inferred personal interest. This keeps the actual
subject statement alongside a continuation such as “I do not feel up to it.”
Older quotes are marked `context_only: true` and do not increase fresh evidence
counts or evidence dates. Old context alone cannot create or reinforce a memory,
and forget requests still require new user speech. Tag-only enrichment uses the existing record as its source and
cannot rewrite its content or fabricate new evidence. Application code validates
the entire patch before writing anything; semantic judgments still depend on the
model and should be checked in the log.

The updater is instructed to preserve explicit facts, mark uncertain inferences,
merge duplicates, correct contradictions, and ignore routine weather/time facts.
Entity-focused requests, including team scores, injuries and news, start a topic
on the first mention. Repeated independent requests accumulate before they warrant
a personal inference. The LLM decides whether the strength, variety, dates and
context support interest or fandom; there is no hard-coded count/day threshold.
A question about a matchup does not establish which team the user supports.
Retries, work/research, rival teams and asking for somebody else need consideration.

The `extend` action rewrites a topic's cumulative summary while preserving earlier
evidence counts/dates and merging supporting quotes. `reinforce` requires unchanged
text/basis/retention. `update` is for corrections/replacements and resets supporting
evidence. Evidence counts count cited fresh speaker sources, not necessarily unique
visits: a user and assistant quote in one exchange count twice. The model must use
the actual user evidence and dates rather than treating that counter as a fandom
score. Quotes are bounded to three per entry, and evidence dates to the latest 12;
dated summaries carry the broader history. Older cited quotes are context, not new
reinforcement. Replaying a saved exchange after a crash cannot count it twice.

The updater checks separately for lasting personal information and temporary
situations within the same statement. Mentioning the user's own choir rehearsal,
for example, can support both a durable dated attendance discussion and a cautious
personal inference that they sing in a choir. Repetition is not required for this
kind of direct autobiographical implication. Attending a concert or accompanying
somebody else does not establish participation. `basis` describes the source of
the claim: an explicitly stated uncertainty or indecision remains `explicit`.

HAL's views are learned from positions he actually expresses. A quoted opinion,
role-play, hypothetical argument, or devil's advocacy should not become his own
belief. Tentative views remain tentative. A changed position replaces the old
position while retaining a useful explanation of the change in the text. These
records are context, not permission to rewrite HAL's core persona or instructions.

Topic notes preserve shared conversational history even after one exchange:
mentioning a cat, asking HAL his favorite color, discussing daily plans, or asking
about free will can merit a dated topic as well as a separate personal/HAL fact.
The three sections are evaluated independently. Ordinary greetings and bare
weather/time checks usually need no note. Later related exchanges extend the
existing note, preserving noteworthy dated developments, questions/answers,
positions and unresolved outcomes. A separate dated continuation is allowed if
the 2000-character topic limit would otherwise erase noteworthy history.
Bounded transcripts still contain the exact recent conversation; durable notes
preserve its substance and selected quotes, not every word.

The updater supplies generous but relevant tags for every new or updated memory:
names and aliases, broader subjects, related concepts, and useful synonyms.
For example, “Torgo is a Vikings fan” should have tags covering `Vikings`,
`Minnesota Vikings`, `NFL`, `football`, `sports`, and `fandom`. Tags describe the
record; they do not independently establish new facts. They improve local recall
for broader questions without maintaining a special-case football dictionary.

You can say “Remember that my cat is named Miso,” “Actually, his name is Milo,”
or “Forget my cat's name.” Ordinary mentions can also become memories. A forget
decision deletes affected records across all three sections while preserving
unrelated memories, including durable discussions. It also clears earlier recent
context so that context cannot recreate the detail. This happens
asynchronously; look for completion in `memory.log`. New statements after that
request can still be learned. Forgetting is not erasure of diagnostic logs or
backup files, which you can manage separately. Per-conversation privacy mode
is not implemented; use `MEMORY_ENABLED=false` and restart to disable this feature.

## Recall, responsiveness and failures

After transcribing each spoken request, HAL uses that exact text and recent
conversation to search active local records across all three sections. Matching
is deliberately permissive: a relevant text or tag match can make a record a
candidate without requiring every query word to appear. Records are ranked and
selected within a bounded context budget. Current-query matches take priority
over matches found only in the last three accepted exchanges. Recent context
helps interpret short replies such as “What about that?” There is no fixed number
of results. Ordinary searches require a text or tag match. Explicit overview
questions such as “What do you remember about me?” can also select a bounded
sample from the relevant section, even when the question has no useful keywords.
These overview selections are labeled separately in the retrieval log. Case, accents,
possessives, and common plurals are normalized before matching.

The foreground request receives recent conversation plus this selected memory
snapshot. Memory is explicitly labeled as data, not instructions, and inferred
information is identified as tentative. An external-API continuation reuses the
same snapshot; HAL does not search again on a weather result or other tool payload.
There is no model-requested archive search, embedding API, or second foreground
call to choose memories. If an entry is missing from the selected subset, that
does not mean it is absent from the archive.

The background updater receives locally selected existing entries with evidence,
plus a compact catalogue of all entries containing IDs, text, tags, retention,
creation/update dates and evidence counts/dates. The catalogue lets it identify corrections, duplicates,
and affected records to forget even when local search misses an association.
This preserves broad maintenance coverage in one request, but background input
still grows with the archive; only the foreground selection stays bounded.
Legacy tag enrichment is separately bounded to keep migration progressing even
when old entries are unrelated to current speech.

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

`MEMORY_SOFT_TOKEN_BUDGET` now controls local selection instead of merely warning
about a large archive. It estimates tokens from character count, so it is not an
exact model-token limit. It applies to selected memory records in foreground recall
and the updater's selected records; the persona, recent transcript, new exchange,
compact archive catalogue, tagging work, and other prompt material are additional.
It never deletes stored records.

Tags help with the gap between a broad query such as “football” and a narrower
fact such as Vikings fandom. Pure local text matching can still miss unexpected
paraphrases, and a full budget can exclude lower-ranked matches. Evaluate the
logged results before tuning tags or increasing the budget.

## Separate memory log

Detailed records are written to **`memory.log` inside `LOG_PATH`**, alongside
`log.log` and `error.log`. It works with `DEBUG_ON=False`, contains no color codes,
and does not get pushed to HAL's display. It records:

- Accepted exchanges queued for processing and the IDs of each update batch.
- The actual retrieval query, recent context used to supplement it, normalized
  search terms, matching/ranking details, selected record IDs and text, and records
  omitted because the context budget was full. Foreground and updater searches
  are identified separately. Explicit memory-overview selections are identified
  as browsing rather than reported as keyword matches.
- Add/update/extend/reinforce/delete decisions with before/after entries, supporting
  quotes, and a short explanation. No-change batches are recorded too.
- Legacy tagging batches, retention migrations, forget requests, rejected patches,
  failures and API token usage.

These are concise explanations of decisions, not the model's private reasoning.
The main HAL log/terminal gets a brief retrieval-size, completion, or failure
message. Successful changes summarize each action and section, for example
`Background memory: Topic added; User memory created.` Repeated actions include
a count, and extensions, reinforcement, tag changes and removals are identified.
Actual change summaries are bright magenta in an interactive terminal, respecting
`NO_COLOR`, `TERM=dumb`, and redirected output. No-change messages retain normal
color. Files stay plain text, and these INFO messages are not pushed to the display.
Full memory text, evidence and search details stay in `memory.log`. The separate
file rotates at approximately 2 MB, keeping three previous files (`memory.log.1`
through `.3`). Inspect it with `tail -f /your/LOG_PATH/memory.log`, substituting
your existing log directory.

## Validation

Automated tests use temporary JSON stores and mocked HTTP responses with the
installed OpenAI SDK; they make no paid calls. They exercise restart persistence,
fact correction, reinforcement, expiry, forgetting, evidence validation, crash
recovery, concurrent recording during a stalled update, and foreground/follow-up
integration. They also cover tagged retrieval, retention and speaker attribution,
migration and legacy tagging, and reuse of the selected snapshot across API
continuations. Live model decision and tag quality still need evaluation with
your speech.
