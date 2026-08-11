---
name: debug
description: Diagnose and debug voice recognition, Silero VAD, wake word detection, LLM output parsing, or TTS engine issues. Use when troubleshooting agent failures or audio pipeline bugs.
---

# Debugging Subsystems

When diagnosing issues in `spych`, follow this systematic subsystem checklist to isolate and resolve root causes without introducing superficial workarounds.

---

## 1. Audio Recording & Silero VAD (`core.py`)

**Symptoms**: Audio cuts off early, fails to stop recording, or transcribes empty strings.

Check points:
- **Silero VAD thresholds**: `spych/core.py` uses hysteresis with separate onset (`threshold`) and offset (`silence_threshold`) parameters.
  - Speech onset default: `0.5`
  - Silence offset default: `0.35`
- **Audio Device Index**: Ensure `device_index` passed to `pvrecorder` matches an active microphone (`-1` selects system default).
- **Duration mode**: `duration=0` or `duration="auto"` activates auto-terminated VAD; numeric `duration > 0` uses fixed recording length.

---

## 2. Wake Word Listener (`wake.py`)

**Symptoms**: Missed wake words, delayed triggers, or thread deadlocks.

Check points:
- **Single Capture Thread**: `SpychWake` runs one persistent `_WakeCapture` thread that owns the mic and calls `Recorder.record_vad()` in a loop — Silero VAD (same hysteresis pattern as `core.py` and `live.py`) blocks it until real speech starts and ends, then the isolated utterance is pushed onto a shared queue. No fixed-duration blind window and no per-cycle device open/close.
- **Transcriber Worker Pool**: `wake_listener_count` (default 2) `_WakeTranscriber` threads pull utterances off that queue and run the wake-spotting Whisper model (`beam_size=1`, greedy) only on isolated speech — never on silence.
- **Model Size**: Wake spotting should always use a small model (`whisper_model`, default `"tiny.en"`). `SpychOrchestrator.build_spych_wake` no longer overrides this to `"base.en"` — if wake detection feels slow, check whether `spych_wake_kwargs` is forcing a heavier model.
- **Locking**: `SpychWake.locked` pauses both the capture thread and transcriber workers while a wake callback is running, so detection doesn't compete with an in-flight response. `stop_event` (a `threading.Event`) is the full-shutdown signal, distinct from `locked`.
- **Wake Word Matching**: `wake_word_map` keys are stripped and lowercased before substring matching against transcribed audio.
- **Terminate Words**: Verify text isn't inadvertently matching entries in `terminate_words`.

---

## 3. LLM Response Parsing (`responders.py`)

**Symptoms**: `json.JSONDecodeError`, empty summary, missing feedback state, or LLM markdown leaks into audio response.

Check points:
- **`format_prompt(user_input)`**: Verifies that the prompt sent to the LLM explicitly requests JSON matching `AgentResponse` schema (`response`, `summary`, `requires_user_feedback`).
- **`parse_output(raw_text)`**: Normalizes markdown code fences (```json ... ```), extracts raw JSON blocks, and falls back gracefully to wrapping unformatted text into `AgentResponse(response=raw_text, summary=raw_text[:120])`.
- Check raw CLI / API output from LLM subprocess to verify output format.

---

## 4. Text-To-Speech Speaker (`speaker/`)

**Symptoms**: Audio output fails, sound stutters, or Chatterbox/Kokoro initialization error.

Check points:
- **Backend Fallback**: `Speaker` (`speaker/speaker.py`) tries `ChatterboxBackend` (primary) first and falls back to `KokoroBackend` or silent printing if TTS dependencies fail.
- **Pygame Playback**: `pygame.mixer` handles audio playback; verify device audio outputs are not muted or locked by another process.
- **Voice Parameters**: Ensure `speaker_voice` string matches available voice models (`af_heart`, `af_bella`, `am_adam`, etc.).

---

## 5. Multi-Agent Orchestrator & CLI Spinner (`orchestrator.py` / `cli_tools.py`)

**Symptoms**: Terminal visual corruption, overlapping spinner text, or garbled output during response.

Check points:
- **Shared Spinner**: `SpychOrchestrator` creates a single `CliSpinner` instance and injects it into all registered `BaseResponder` instances.
- **Output Locks**: `CliPrinter` suppresses spinner updates while printing agent responses to prevent terminal frame interleaving.

---

## 6. Diagnostic Logging Policy

- Always inspect full stack traces before attempting code edits.
- Never wrap broken logic in silent `try/except: pass` blocks or return fake empty arrays to hide errors.
- Rerun tests ([test](../test/SKILL.md)) to confirm fixes.
