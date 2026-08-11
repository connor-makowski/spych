---
name: live-pipeline
description: Develop or modify continuous live transcription (SpychLive) or bilingual live translation (SpychLiveTranslation). Use when working on live audio stream processing, subtitle generation, or Ollama translation.
---

# Live Transcription & Translation Pipelines

`spych` provides continuous, streaming audio processing pipelines for recording live speech directly to disk and translating speech in real time.

---

## 1. Live Transcription Architecture (`spych/live.py`)

`SpychLive` uses a multi-threaded producer-consumer model:

```
VADRecorder (Thread) ---> Queue ---> Transcriber (Thread) ---> Queue ---> Writer (Thread)
```

- **`VADRecorder`**: Continuously listens to audio input and emits speech chunks split by VAD silences.
- **`Transcriber`**: Runs `faster-whisper` transcription on incoming chunks. Maintains a context window (~128 words) supplied to Whisper's `initial_prompt` for context preservation.
- **`Writer`**: Writes formatted text chunks into `.txt` and/or `.srt` subtitle files with precise timestamps.

---

## 2. Live Translation Architecture (`spych/live_translation.py`)

`SpychLiveTranslation` extends the live pipeline by injecting Ollama translation:

```
VADRecorder ---> TranslatingTranscriber (Whisper + Ollama) ---> TranslationWriter (Disk + Speaker)
```

- **Whisper Model Name Normalization**: Automatically strips `.en` suffix from model names (e.g. `base.en` -> `base`) when `input_language != "en"`.
- **Bilingual Formatting**: `TranslationWriter` outputs dual-language timestamps:
  `[00:01:23] (es) Hola mundo`
  `[00:01:23] (en) Hello world`
- **Fallback Resilience**: If Ollama server is unreachable or fails to respond, translation falls back to `[translation unavailable]` without halting transcription or crashing the audio loop.
- **TTS Integration**: Translated text can optionally be spoken aloud via `Speaker`.

---

## 3. Modifying & Testing Live Pipelines

When modifying `live.py` or `live_translation.py`:
1. Ensure thread lock safety when appending to file write queues or context buffers.
2. Verify clean thread cleanup when terminating via keystroke, spoken terminate word, or `SIGINT`.
3. Test using `test/11_translation.py` or running the CLI:
   ```bash
   spych live --output transcript.srt
   spych translate --target-lang es
   ```
4. Run [prettify](../prettify/SKILL.md).
