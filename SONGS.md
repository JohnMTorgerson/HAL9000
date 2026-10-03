# Singing Daisy Bell

Ask HAL to sing a song, sing Daisy, or sing it again. The LLM decides whether
the request is for a performance and returns a local command:

```text
[PLAY_SONG] {"song": "daisy", "intro": "Certainly."}
```

HAL speaks the brief introduction, pauses for one second, and plays
`src/HAL-clips/Daisy.wav` through the configured output device. The recording
is the approved loudness-matched version, unchanged. It bypasses voice
high-pass filtering and peak normalization; output-device resampling still
applies when needed. No new dependencies or environment settings are required.

The command works for explicit wake/spacebar requests and accepted follow-ups.
The microphone stays closed through the introduction, pause, and entire song.
If follow-ups are enabled, their window starts after playback subject to the
existing conversation session limit. Ctrl-C still stops HAL.

Only Daisy is bundled. Questions about the song, quoted singing requests, and
requests not to sing should receive ordinary replies. A request for another
specific song should receive an offer to sing Daisy instead. These intent
decisions are made by the LLM and need normal microphone testing.

Malformed commands or a missing clip receive a spoken failure response.
Playback/device errors are logged and displayed, then listening resumes.
Conversation history and optional persistent dialogue record the introduction
plus a clearly labeled application result, including failures. The command
itself and song lyrics are never sent to speech synthesis.
