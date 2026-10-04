# Image lookup

HAL can find an existing picture, check its relevance with vision, and display
it in the upper panel. Clickable source citations sit in the bottom-right corner
of the image panel on a translucent dark background. The visible links show only
the source domains (without `www.`); each still opens the full source webpage.
The lower conversation panel expires normally, independently of the image.
The selected source webpage and original image URL are also written to `log.log`
as `IMAGE SHOWN` / `IMAGE CITATION` entries.

## Install the update

Extract the update at the repository root, preserving its `src/` directory:
The archive also includes the earlier Daisy update, so it can be applied whether
or not that ZIP was already extracted.

```bash
unzip -o "$HOME/Downloads/HAL9000-images-update.zip" -d "/Volumes/2TB-SSD/Coding/HAL9000"
```

Using the Python environment you normally use to run HAL, install the small
image dependency:

```bash
python -m pip install -r "/Volumes/2TB-SSD/Coding/HAL9000/src/requirements-images.txt"
```

Neither command starts HAL. On a Pi, use its repository path instead. Restart
HAL and its display server when you are ready, then reload the display browser
to load the updated JavaScript and CSS. If the display server runs separately
on another machine, update its display files and install the image dependency
there too. The image client uploads the selected picture to that server.

Your `.env` is not replaced. Image lookup uses the existing `LLM_BACKEND`,
`LLM_MODEL`, `OPENAI_API_KEY`, `LLM_SERVICE_TIER`, and `DISPLAY_SERVER_URL` settings.
The OpenAI model needs Responses API image-search, image-input, and structured
output support (the planned configuration is `gpt-6-luna`). The implementation
works with the project's pinned OpenAI Python SDK 1.99.9; no SDK upgrade is needed.
An unsupported model/tool configuration produces a spoken failure, not a crash.

## Try it

- “Hey HAL, what does the new iPhone look like?”
- “Hey HAL, show me a picture of a 1968 Mustang fastback.”
- “Show me another one.”
- “Go back to the previous picture.”
- “Show me the convertible instead.”
- “Close that picture.”

Follow-ups without “Hey HAL” require the existing follow-up listening option.
Otherwise use the wake phrase or push-to-talk for each command. A changed model,
color, angle, or variant starts a new lookup; next/previous reuse checked images.
At the end of the cached list HAL says there is no further image in that direction.

## Request flow and limits

1. The ordinary HAL conversation request emits an image action.
2. The OpenAI image provider uses one Responses request to resolve the subject
   from current sources and find image candidates. It can make up to three
   hosted search-tool calls inside that request. URLs come from structured tool
   results, never URLs invented in the assistant's prose.
3. Up to five candidate downloads run with bounded sizes and timeouts. Duplicate
   pictures are removed. Up to three usable images are sent together in one
   vision request, with captions and source evidence. That request chooses and
   describes the acceptable images, including the ready-to-speak HAL reply.
4. The selected checked image is uploaded to the display's local cache. The
   browser confirms that it loaded before HAL speaks the success reply. A broken
   image can fall back to another checked candidate; an unavailable browser ends
   the attempt with a spoken display failure.

The ordinary successful path therefore uses three LLM requests total, including
the initial HAL turn. There is no separate final rephrasing or moderation call.
Search-tool calls have their own cost in addition to model usage. There are no
automatic LLM retries. Search and vision requests each have a 45-second network
timeout; the browser acknowledgment wait is eight seconds.

Images use centered `cover` sizing: they fill the panel without stretching, with
edges cropped when the aspect ratios differ. They stay visible for two minutes.
The image and its citation disappear together. Checked alternatives remain in memory for twenty
minutes, or until a new successful search or explicit close. The display's disk
cache retains at most twenty images and removes pictures older than a day when
the next image is shown. It is excluded from Git and from update archives.

Provider refusals are separate from empty results, unsupported backends, and
technical failures. HAL speaks the provider's explanation when available and
does not invent a policy category when no reason was supplied. Native refusals,
content-filter responses, and structured refusals are handled before displaying
an image. Refusals are logged and are not retried through another provider.
HAL adds no content blacklist or independent moderation service.

With `LLM_BACKEND=ollama`, image actions explain that image lookup is unavailable
with Ollama. They do not create an OpenAI image client or make OpenAI image calls.
The provider interface also gives future backends a place to implement lookup
without changing display, caching, or conversation handling. Video is not included.

## Validation

Automated tests cover the pinned SDK's HTTP payloads and response decoding using
mocked API responses; source handling; batch vision selection; refusals at each
stage; download/display failures; navigation; cache expiration; citation lifetime;
unsupported backends; and the actual voice-loop function without audio hardware.
The optional browser test exercises the real display server, browser load
acknowledgment, and clickable citations (`HAL_BROWSER_TESTS=1`, with Playwright
and Chromium installed). Live search quality and account/model availability still
need a first run with your own API account.

## Image layout update

The update moves image citations from the conversation panel onto the image and
switches image fitting to centered `cover` sizing. There are no new dependencies.
After updating the display files, restart the display server and reload the
display browser to apply the layout change.

References:
- https://developers.openai.com/api/docs/guides/tools-web-search
- https://developers.openai.com/api/docs/guides/images-vision
- https://developers.openai.com/api/docs/guides/structured-outputs
