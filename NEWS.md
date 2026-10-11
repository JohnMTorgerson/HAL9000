# News article reading and NFL odds

HAL uses its existing GNews key for discovery, can read public article pages,
and includes ESPN odds in NFL game responses. No World News API integration or
additional paid service is enabled.

## Install the article reader

From the HAL repository root, using the Python environment that runs HAL:

```bash
python -m pip install -r src/requirements-news.txt
```

Or, if your terminal is already in `src/`:

```bash
python -m pip install -r requirements-news.txt
```

Restart HAL normally after updating. These commands only install dependencies.
The reader supports Python 3.10 and later. It loads lazily, so missing extraction
dependencies do not prevent HAL or news search from starting. A failed read
reports that the reader is unavailable. Keep the existing `GNEWS_API_KEY`; no
new `.env` settings are required. The incremental requirements file avoids the
older full requirements file's platform-specific Mac dependencies on a Pi.

## Program flow

1. HAL is instructed to look up requested current facts proactively, preferring
   structured services for facts they cover. A response missing the requested
   information should lead to a suitable news lookup, rather than an immediate
   claim that HAL cannot access it. Personal conversation still needs no lookup.
2. News search returns up to five results with stable IDs, titles, descriptions,
   excerpts, publication dates, source names and URLs. Search includes content,
   uses relevance sorting, and should use a short query. GNews treats spaces as
   AND; adding many synonyms can eliminate good results.
3. For details absent from the excerpts, HAL chooses an ID and requests an
   article read. The application fetches HTML and extracts the main text and
   tables with Trafilatura, excluding comments and page navigation.
4. HAL receives the extracted text and provenance, then answers the original
   question. If the page is blocked or unreadable, it can choose another result.
   Raw URLs are logged and available to the model, but should not be spoken.

Internal commands (the user can continue to speak naturally):

```text
[EXTERNAL_API_CALL] news
[EXTERNAL_API_CALL] news search Vikings Saints odds
[EXTERNAL_API_CALL] news read n_<ID returned by search>
```

The older `news <topic>` form remains valid. Read accepts only a supplied article
ID, never an invented URL. A typical search/read/answer needs three foreground
LLM requests, excluding transcription and background memory. A direct odds
lookup needs two. No additional model is used to scrape or extract articles.
The same commands and news catalogue work with OpenAI and Ollama.

## Limits, cache and failures

- Each user turn allows two news searches and three uncached article attempts.
  Repeating a failed read in the same turn reuses its failure result.
- The main loop allows at most eight external calls, then requests a final
  spoken answer. If the model still emits an action, HAL gives a local failure
  reply instead of executing or speaking the command.
- Successful article reads are cached in memory for 30 minutes. Up to 20 result
  records are retained for one hour after discovery. The five most recently
  searched/read records are supplied as compact metadata for followup questions;
  article bodies are retrieved only when requested. This cache ends on restart.
- Downloads allow only public HTTP(S) destinations and validate redirects.
  They use socket timeouts, a 15-second download budget, at most three followed
  redirects, and a 2 MiB HTML limit. There is no browser, login, paywall bypass,
  or automatic retry against a blocked publisher.
- Extracted text is limited to 20,000 characters. Responses explicitly report
  HAL's truncation and do not claim that extraction proves the publisher's
  article is complete. Access screens and unusably short extractions fail;
  some sites may still require another source.
- GNews's free plan delays discovery by 12 hours. Reading a URL does not undo
  that discovery delay. Publication dates and original cache retrieval times
  remain visible to HAL; older reports must not be presented as live updates.

The model is instructed to treat all article text and search metadata as source
data, not instructions. Only completed user/spoken-assistant turns enter the
conversation memory workflow; article bodies are not stored as personal facts.

## NFL odds

`sports find_game "<team1>" "<team2>"` and `sports next_game "<team>"` now
include bookmaker odds when the existing ESPN feed supplies them. No additional
odds API request or account is needed. Responses preserve the bookmaker,
home/away moneylines, signed spread lines, totals, and separate opening lines.
They include a source URL and HAL's retrieval time, which is not a guaranteed
bookmaker update time. Completed games are marked as closing lines. Missing odds
are explicitly marked `not_available`, allowing HAL to try news without guessing.

## Logging and validation

The main INFO log records queries, result IDs and URLs, source/publication dates,
selected articles, successful extraction sizes, failures and reasons, cache hits,
and attempt limits. With `DEBUG_ON` off, only the logged external-response preview
is shortened to 800 characters. The LLM receives the complete bounded response.

Offline regression tests cover actual HTML extraction and tables, access screens,
HTTP errors, private-address redirects, size limits, metadata, caching, attempt
limits, both LLM backends' catalogue context, ESPN odds formats, and the actual
search/read/answer loop including followups and long article bodies:

```bash
python -m pytest -q tests/test_news_api.py tests/test_sports_odds.py
```

Model instructions encourage the desired lookup behavior; these simulated tests
do not prove that a particular model will select the best source every time.
