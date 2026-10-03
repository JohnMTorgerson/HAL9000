"""The LLM's local song command; only the bundled Daisy recording is playable."""
from dataclasses import dataclass
import json
from pathlib import Path


PLAY_SONG_MARKER = '[PLAY_SONG]'
DAISY_PATH = Path(__file__).resolve().parent / 'HAL-clips' / 'Daisy.wav'
SONG_PAUSE_SECONDS = 1.0
SONG_FAILURE_REPLY = "I'm sorry, I couldn't play the song."


class SongRequestError(ValueError):
    """A song command was recognized but could not safely be interpreted."""


@dataclass(frozen=True)
class SongRequest:
    intro: str


def parse_song_request(reply):
    """Return None for ordinary speech, or validate a complete song command."""
    reply = reply.strip()
    if not reply.startswith(PLAY_SONG_MARKER):
        return None
    try:
        payload = json.loads(reply[len(PLAY_SONG_MARKER):].strip())
    except (TypeError, ValueError):
        raise SongRequestError('Expected JSON after [PLAY_SONG].') from None
    if (not isinstance(payload, dict) or set(payload) != {'song', 'intro'} or
            payload['song'] != 'daisy' or not isinstance(payload['intro'], str)):
        raise SongRequestError('Only the bundled daisy song and a spoken intro are supported.')
    intro = payload['intro'].strip()
    if not intro or len(intro) > 200 or any(char in intro for char in '[]\r\n'):
        raise SongRequestError('The song introduction must be a short, plain spoken line.')
    return SongRequest(intro=intro)
