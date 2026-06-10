"""
# Spych
[![PyPI version](https://badge.fury.io/py/spych.svg)](https://badge.fury.io/py/spych)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI Downloads](https://static.pepy.tech/badge/spych/month)](https://pepy.tech/project/spych)

**Spych** (pronounced "speech"): Talk with your computer like it's your personal assistant without sending your voice to the cloud.

A lightweight, fully offline Python toolkit for wake word detection, audio transcription, spoken AI responses, and AI integrations. Built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [PvRecorder](https://github.com/Picovoice/pvrecorder), and [Kokoro](https://github.com/hexgrad/kokoro).

**API Docs**: https://connor-makowski.github.io/spych/spych.html

---

# Installation

### Recommended: uv

```bash
uv tool install spych
```

### Alternative: pip

```bash
pip install spych
```

### TTS Extras

By default, Spych automatically installs the right TTS backend for your Python version. You can also install explicitly:

```bash
uv tool install "spych[kokoro]"       # Fast, lightweight (Python < 3.13 recommended)
uv tool install "spych[chatterbox]"   # High-quality voice cloning (Python >= 3.13 required)
```

---

# Quick Start

```bash
# Navigate to your project directory first
cd ~/my_project

# Voice-control Claude Code — say "hey claude" to trigger
spych claude

# Use a personality preset — say "hey jarvis" to trigger
spych claude --personality jarvis

# Voice-control a local Ollama model — say "hey llama" to trigger
spych ollama --model llama3.2:latest
```

> 💡 **Pro tip:** Saying "Hey Claude" or "Hey Llama" tends to trigger more reliably than the bare wake word.

Say **"terminate"** (or press `Ctrl+C`) to stop any session.

---

# CLI

## Available Agents

All agents require their respective CLI tool to be installed and authenticated before use.

| Command | Alias | Description | Default wake words |
|---|---|---|---|
| `spych claude_code_cli` | — | Voice-control Claude Code via the CLI | `claude`, `clod`, `cloud`, `clawed` |
| `spych claude_code_sdk` | `spych claude` | Voice-control Claude Code via the Agent SDK | `claude`, `clod`, `cloud`, `clawed` |
| `spych codex_cli` | `spych codex` | Voice-control the OpenAI Codex agent | `codex` |
| `spych gemini_cli` | `spych gemini` | Voice-control the Google Gemini agent | `gemini`, `google` |
| `spych opencode_cli` | `spych opencode` | Voice-control the OpenCode agent | `opencode`, `open code` |
| `spych ollama` | — | Talk to a local Ollama model | `llama`, `ollama`, `lama` |

## Available Utilities

The following utilities are also available as CLI commands. They don't use wake words, but serve various auxiliary functions like live transcription and voice profiling.

| Command | Description |
|---|---|
| `spych --version` | Print the version number and exit |
| `spych --help` | Show detailed usage instructions and exit |
| `spych live` | Continuous speech-to-text transcription to file |
| `spych live-translation` | (Beta Mode) Continuous speech transcription + translation (bilingual input and output) |
| `spych multi` | Run multiple agents simultaneously |
| `spych users` | Manage user profiles and global settings |
| `spych profile_my_voice` | Record a voice sample for TTS cloning |

## Global Flags

These must be placed **before** the agent name:

```bash
spych --theme light claude
```

| Flag | Options | Default | Description |
|---|---|---|---|
| `--theme` | `dark`, `light`, `solarized`, `mono` | `dark` | Terminal colour theme |

> 💡 **TUI Dashboard:** Spych launches a rich terminal interface by default. Use the `--verbose` flag (e.g., `spych --verbose claude`) to switch to a simpler, non-interactive scrollable output.

---

## Common Flags

All agent subcommands accept these flags:

| Flag | Default | Description |
|---|---|---|
| `--personality NAME` | — | Apply a named preset (sets wake words, voice, name, style) |
| `--name NAME` | *(agent default)* | Custom display name shown in the terminal |
| `--wake-words WORD [...]` | *(agent default)* | One or more words that trigger the agent |
| `--terminate-words WORD [...]` | `terminate` | Words that stop the listener |
| `--listen-duration SECONDS` | `0` (VAD auto) | Seconds to record after wake word |
| `--follow-up-listen-duration SECONDS` | `0` | Seconds to listen for a follow-up answer |
| `--inactivity-timeout SECONDS` | `4.0` | Seconds of silence before returning to wake word |
| `--use-speaker BOOL` | `true` | Speak responses aloud via TTS |
| `--speaker-voice VOICE` | `af_heart` | Voice name for spoken responses |
| `--speaker-backend BACKEND` | *(auto)* | `chatterbox` or `kokoro` |
| `--response-style STYLE` | — | Style preset or custom instruction for spoken output |
| `--intermediate-responses BOOL` | `true` | Enable intermediate response chaining for long-running tasks |

Coding agents (`claude`, `codex`, `gemini`, `opencode`) also accept:

| Flag | Default | Description |
|---|---|---|
| `--continue-conversation BOOL` | `true` | Resume the most recent session |
| `--show-tool-events BOOL` | `true` | Print live tool start/end events |

Agent-specific flags:

| Agent | Flag | Default | Description |
|---|---|---|---|
| `ollama` | `--model` | `llama3.2:latest` | Ollama model name |
| `ollama` | `--history-length` | `10` | Past interactions to include in context |
| `ollama` | `--host` | `http://localhost:11434` | Ollama instance URL |
| `opencode_cli` | `--model` | — | Model in `provider/model` format |
| `claude_code_sdk` | `--setting-sources` | `user project local` | Claude Code settings sources |

---

# Personalities

Personalities are named presets that bundle a wake word list, voice, display name, and response style into a single flag. Any explicit flag overrides the preset.

```bash
spych claude --personality jarvis
# equivalent to:
spych claude --name "JARVIS" --wake-words jarvis jarves \
             --speaker-voice bm_george --use-speaker true \
             --response-style jarvis
```

| Name | Wake words | Voice | Style |
|---|---|---|---|
| `assistant` | `assistant`, `helper`, `computer` | `af_heart` | `assistant` — helpful, precise, informative |
| `friend` | `friend`, `buddy`, `pal` | `af_amy` | `friendly` — warm and simple |
| `jarvis` | `jarvis`, `jarves`, `jargus`, `jervis` | `bm_george` | `jarvis` — precise, dry wit, "sir" |
| `pirate` | `blackbeard`, `pirate`, `ahoy` | `am_michael` | `pirate` — pirate speak, colorful |
| `news_anchor` | `bella`, `news anchor`, `anchor` | `af_bella` | `news_anchor` — professional broadcast tone |
| `robot` | `rob`, `robot` | `am_adam` | `robot` — monotone, literal |
| `caveman` | `er`, `ur`, `caveman`, `cave man` | `am_onyx` | `caveman` — very simple, direct |

---

# User Management

Spych supports multiple user profiles, allowing agents to provide more personalized responses based on your name, age, and other context.

```bash
# Launch the interactive user management menu
spych users
```

The `users` utility allows you to:
- Create, edit, and delete user profiles.
- Set a default user for all agents.
- Change the global terminal theme (`dark`, `light`, `solarized`, `mono`).

You can also specify a user for a specific session:
```bash
spych claude --user Connor
```

---

# Response Styles

The `--response-style` flag shapes how the agent formats its spoken output.

| Style | Description |
|---|---|
| `assistant` | Helpful and precise, concise and informative |
| `concise` | Key points only, direct |
| `friendly` | Warm, approachable, simple language |
| `military` | Brevity-style, short sentences |
| `five_year_old` | Simple words, very short |
| `fast` | As brief as reasonably possible |
| `pirate` | Pirate speak, colorful |
| `news_anchor` | Professional broadcast tone |
| `haiku` | 5-7-5 haiku form |
| `shakespearean` | Elizabethan English |
| `robot` | Monotone, literal |
| `caveman` | Very simple, direct |
| `yoda` | Inverted sentence structure |
| `jarvis` | JARVIS from Iron Man — precise, dry wit, addresses user as "sir" or "ma'am" |

You can also pass any custom instruction string directly: `--response-style "Reply in exactly one sentence."`.

---

# Text-to-Speech & Voices

Spoken responses are enabled by default for personality presets and when `--use-speaker true` is set.

```bash
spych claude --use-speaker true --speaker-voice bm_george
spych claude --use-speaker true --speaker-backend kokoro
spych claude --use-speaker false   # disable TTS
```

When TTS is active, short responses are spoken verbatim; longer ones use the agent's short `summary`. If the response ends with a question, Spych automatically listens for a follow-up — no wake word required.

### TTS Backends

| Backend | Best for | Python support |
|---|---|---|
| **Chatterbox** (default priority) | Natural voices, zero-shot voice cloning | 3.11+ (required for 3.13+) |
| **Kokoro** (lightweight fallback) | Fast, low-resource devices (e.g. Raspberry Pi) | 3.11–3.12 recommended |

Spych tries Chatterbox first, then Kokoro. Use `--speaker-backend` to force one explicitly.

### Available Voices

The same voice names work for both backends.

- Chatterbox wave voices: https://github.com/connor-makowski/spych/tree/main/voices/wave
- Kokoro pt voices (56 total): https://github.com/connor-makowski/spych/tree/main/voices/pt

American English (`am_` / `af_`):

| Voice | Gender | Grade |
|---|---|---|
| `af_heart` | F | A (default) |
| `af_bella` | F | A- |
| `af_nicole` | F | B- |
| `am_michael` | M | C+ |
| `am_fenrir` | M | C+ |
| `am_puck` | M | C+ |

British English (`bm_` / `bf_`):

| Voice | Gender | Grade |
|---|---|---|
| `bf_emma` | F | B- |
| `bf_isabella` | F | C |
| `bm_george` | M | C |

### Voice Cloning

Record a 10-second sample of your voice, then use it as the speaker voice. Requires the Chatterbox backend.

```bash
# Step 1: record your profile
spych profile_my_voice --name my_voice

# Step 2: use it
spych claude --use-speaker true --speaker-voice my_voice --speaker-backend chatterbox

# Or use any .wav file directly
spych claude --use-speaker true --speaker-voice /path/to/my_voice.wav --speaker-backend chatterbox
```

---

# Live Transcription

`spych live` continuously records from the microphone using VAD and writes the transcript to disk in real time. No wake word required — it transcribes everything until stopped.

## CLI

```bash
spych live                                                 # writes transcript.srt
spych live --output-path meeting --output-format both
spych live --terminate-words "stop recording"
spych live --no-timestamps --whisper-model medium.en
```

Stop by pressing the stop key (default: `q` + Enter), saying a terminate word, or pressing `Ctrl+C`.

### Parameters

| Flag | Default | Description |
|---|---|---|
| `--output-path PATH` | `transcript` | Base output file path without extension |
| `--output-format FORMAT` | `srt` | `txt`, `srt`, or `both` |
| `--no-timestamps` | false | Omit timestamps from terminal and `.txt` output |
| `--stop-key KEY` | `q` | Key (then Enter) to stop the session |
| `--terminate-words WORD [...]` | — | Spoken words that stop the session |
| `--device-index N` | `-1` | Microphone device index; -1 uses system default |
| `--whisper-model MODEL` | `base.en` | faster-whisper model name |
| `--whisper-device DEVICE` | `cpu` | `cpu` or `cuda` |
| `--whisper-compute-type TYPE` | `int8` | `int8`, `float16`, or `float32` |
| `--no-speech-threshold FLOAT` | `0.3` | Whisper segments above this `no_speech_prob` are dropped |
| `--speech-threshold FLOAT` | `0.5` | VAD speech onset probability |
| `--silence-threshold FLOAT` | `0.35` | VAD silence probability during speech |
| `--silence-frames N` | `20` | Consecutive silent frames to end a segment (~32ms each) |
| `--speech-pad-frames N` | `5` | Pre-roll frames and onset confirmation count |
| `--max-speech-duration SECONDS` | `30.0` | Hard cap on a single segment |
| `--context-words N` | `32` | Trailing words passed as whisper `initial_prompt` |

## Python

```python
from spych.live import SpychLive

SpychLive(
    output_format="srt",          # "txt", "srt", or "both"
    output_path="my_transcript",  # written to my_transcript.srt
    show_timestamps=True,
    stop_key="q",                 # type q + Enter to stop
    terminate_words=["stop recording"],
).start()
```

### `SpychLive` Parameters

| Parameter | Default | Description |
|---|---|---|
| `output_format` | `"srt"` | Output format(s): `"txt"`, `"srt"`, or `"both"` |
| `output_path` | `"transcript"` | Base path without extension |
| `show_timestamps` | `True` | Prepend `[HH:MM:SS]` timestamps to terminal and `.txt` output |
| `stop_key` | `"q"` | Key (then Enter) to stop the session |
| `terminate_words` | `None` | Spoken words that stop the session |
| `on_terminate` | `None` | No-argument callback executed when a terminate word fires |
| `device_index` | `-1` | Microphone device index; `-1` uses system default |
| `whisper_model` | `"base.en"` | faster-whisper model name |
| `whisper_device` | `"cpu"` | Device for inference: `"cpu"` or `"cuda"` |
| `whisper_compute_type` | `"int8"` | Compute precision: `"int8"`, `"float16"`, or `"float32"` |
| `no_speech_threshold` | `0.4` | Whisper segments above this are discarded |
| `speech_threshold` | `0.5` | Silero VAD onset probability |
| `silence_threshold` | `0.35` | Silero VAD silence probability during speech |
| `silence_frames_threshold` | `20` | Consecutive silent frames to close a segment |
| `speech_pad_frames` | `5` | Pre-roll frame count and onset confirmation threshold |
| `max_speech_duration_s` | `30.0` | Hard cap on a single segment in seconds |
| `context_words` | `32` | Trailing transcript words passed as `initial_prompt` |

---

# Live Translation

**Currently in beta: expect some rough edges.**

`spych live-translation` starts a **bidirectional** live translation session between two languages. Either participant can speak in either language — Whisper transcribes each utterance, Ollama detects which language was spoken and translates it to the other, and each segment is shown as two lines in real time. The translated text is also spoken aloud via TTS by default.

Note: This is currently in beta mode and the API may change without a major version bump. Feedback is very welcome!

Requires [Ollama](https://ollama.com) running locally with the translation model pulled. By default, Spych uses the `llama3.2` model, but you can specify any Ollama model with `--ollama-translation-model`.

## CLI

```bash
# English ↔ Spanish conversation
spych live-translation --languages en es

# English ↔ French, disable TTS
spych live-translation --languages en fr --no-speaker

# Use a larger Ollama model and save a bilingual transcript
spych live-translation --languages en de \
    --ollama-translation-model mistral --output-format both
```

Stop by pressing the stop key (default: `q` + Enter), saying a terminate word, or pressing `Ctrl+C`.

Each utterance is shown as two lines — the original and the translation:

```
[00:00:05](es) Hola, ¿cómo estás?
[00:00:05](en) Hello, how are you?
```

If Ollama is unreachable, the session continues and the translation line shows `[translation unavailable]` for that segment.

### Parameters

| Flag | Default | Description |
|---|---|---|
| `--languages LANG LANG` | *(required)* | Two BCP-47 language codes for the conversation pair (e.g. `en es`) |
| `--ollama-host URL` | `http://localhost:11434` | Ollama HTTP base URL |
| `--ollama-translation-model MODEL` | `llama3.2` | Ollama model used for translation |
| `--no-speaker` | false | Disable TTS — speaker is on by default |
| `--speaker-voice VOICE` | *(model default)* | Wave voice name for zero-shot cloning; omit to use the model's built-in default voice |
| `--output-path PATH` | `transcript` | Base output file path without extension |
| `--output-format FORMAT` | *(none)* | Save transcript to file: `txt`, `srt`, or `both`; omit for terminal-only output |
| `--no-timestamps` | false | Omit timestamps from terminal and `.txt` output |
| `--stop-key KEY` | `q` | Key (then Enter) to stop the session |
| `--terminate-words WORD [...]` | — | Spoken words that stop the session |
| `--device-index N` | `-1` | Microphone device index; -1 uses system default |
| `--whisper-model MODEL` | `small` | faster-whisper model name; `.en` suffix is stripped automatically |
| `--whisper-device DEVICE` | `auto` | `auto`, `cpu`, or `cuda`; `auto` selects `cuda` when available on Python ≤3.13 |
| `--whisper-compute-type TYPE` | `int8` | `int8`, `float16`, or `float32` |
| `--no-speech-threshold FLOAT` | `0.3` | Whisper segments above this `no_speech_prob` are dropped |
| `--speech-threshold FLOAT` | `0.5` | VAD speech onset probability |
| `--silence-threshold FLOAT` | `0.35` | VAD silence probability during speech |
| `--silence-frames N` | `20` | Consecutive silent frames to end a segment (~32ms each) |
| `--speech-pad-frames N` | `5` | Pre-roll frames and onset confirmation count |
| `--max-speech-duration SECONDS` | `30.0` | Hard cap on a single segment |

## Python

```python
from spych.live_translation import SpychLiveTranslation

SpychLiveTranslation(
    lang_a="en",
    lang_b="es",
    output_format="both",          # "txt", "srt", or "both"; "" for terminal only
    output_path="my_translation",  # written to my_translation.txt + .srt
    show_timestamps=True,
    stop_key="q",
    terminate_words=["terminate"],
    ollama_host="http://localhost:11434",
    ollama_translation_model="llama3.2",
    use_speaker=True,
    speaker_voice="",              # "" uses the model's built-in default voice
).start()
```

### `SpychLiveTranslation` Parameters

| Parameter | Default | Description |
|---|---|---|
| `lang_a` | *(required)* | BCP-47 code of the first language (e.g. `"en"`) |
| `lang_b` | *(required)* | BCP-47 code of the second language (e.g. `"es"`) |
| `output_format` | `""` | Output format(s): `"txt"`, `"srt"`, or `"both"`; `""` disables file output |
| `output_path` | `"transcript"` | Base path without extension |
| `show_timestamps` | `True` | Prepend `[HH:MM:SS]` timestamps to terminal and `.txt` output |
| `stop_key` | `"q"` | Key (then Enter) to stop the session |
| `terminate_words` | `None` | Spoken words that stop the session |
| `device_index` | `-1` | Microphone device index; `-1` uses system default |
| `whisper_model` | `"base"` | faster-whisper model name; `.en` suffix stripped automatically |
| `whisper_device` | `"auto"` | Device for inference: `"auto"`, `"cpu"`, or `"cuda"` |
| `whisper_compute_type` | `"int8"` | Compute precision: `"int8"`, `"float16"`, or `"float32"` |
| `no_speech_threshold` | `0.4` | Whisper segments above this are discarded |
| `speech_threshold` | `0.5` | Silero VAD onset probability |
| `silence_threshold` | `0.35` | Silero VAD silence probability during speech |
| `silence_frames_threshold` | `20` | Consecutive silent frames to close a segment |
| `speech_pad_frames` | `5` | Pre-roll frame count and onset confirmation threshold |
| `max_speech_duration_s` | `30.0` | Hard cap on a single segment in seconds |
| `ollama_host` | `"http://localhost:11434"` | Ollama HTTP base URL for translation |
| `ollama_translation_model` | `"llama3.2"` | Ollama model used for translation |
| `use_speaker` | `True` | Speak each translated segment aloud via TTS |
| `speaker_voice` | `""` | Wave voice name for zero-shot cloning; `""` uses the model's built-in default voice |

---

# Multi-agent

Run several agents simultaneously under a single listener, each bound to its own wake words. Say "hey claude" to talk to Claude, "hey llama" to talk to Ollama — all in the same terminal session.

## CLI

```bash
# Two agents, default wake words
spych multi --agents claude gemini

# Include Ollama with a specific model
spych multi --agents claude ollama --ollama-model llama3.2:latest

# Tune listen duration across all agents
spych multi --agents claude codex --listen-duration 8
```

### Multi-agent CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--agents AGENT [...]` | *(required)* | Agents to run: `claude` (`claude_code_cli`), `claude_sdk` (`claude_code_sdk`), `codex` (`codex_cli`), `gemini` (`gemini_cli`), `opencode` (`opencode_cli`), `ollama` |
| `--terminate-words WORD [...]` | `terminate` | Words that stop all agents |
| `--listen-duration SECONDS` | `5` | Seconds to listen after a wake word |
| `--follow-up-listen-duration SECONDS` | `0` | Seconds to listen for follow-up answers |
| `--inactivity-timeout SECONDS` | `4.0` | Seconds of silence before returning to wake word |
| `--continue-conversation BOOL` | `true` | Resume the most recent session for each coding agent |
| `--show-tool-events BOOL` | `true` | Print live tool start/end events |
| `--use-speaker BOOL` | `true` | Speak responses aloud via TTS |
| `--speaker-backend BACKEND` | *(auto)* | `chatterbox` or `kokoro` |
| `--intermediate-responses BOOL` | `true` | Enable intermediate response chaining for long-running tasks |
| `--ollama-model MODEL` | `llama3.2:latest` | Only used when `ollama` is in `--agents` |
| `--ollama-host URL` | `http://localhost:11434` | Only used when `ollama` is in `--agents` |
| `--ollama-history-length N` | `10` | Only used when `ollama` is in `--agents` |
| `--opencode-model MODEL` | — | `provider/model` format. Only used when `opencode_cli` is in `--agents` |
| `--setting-sources SOURCE [...]` | `user project local` | Only used when `claude_code_sdk` is in `--agents` |

## Python

```python
from spych.core import Spych
from spych.orchestrator import SpychOrchestrator
from spych.agents.claude import LocalClaudeCodeCLIResponder
from spych.agents.ollama import OllamaResponder

spych_object = Spych(whisper_model="base.en")

SpychOrchestrator(
    entries=[
        {
            "responder": LocalClaudeCodeCLIResponder(spych_object=spych_object),
            "wake_words": ["claude", "clod", "cloud", "clawed"],
            "terminate_words": ["terminate"],
        },
        {
            "responder": OllamaResponder(spych_object=spych_object, model="llama3.2:latest"),
            "wake_words": ["llama", "ollama", "lama"],
        },
    ]
).start()
```

### `OrchestratorEntry` Keys

| Key | Required | Default | Description |
|---|---|---|---|
| `responder` | ✓ | — | A `BaseResponder` instance |
| `wake_words` | ✓ | — | Words that trigger this responder. Must be unique across all entries |
| `terminate_words` | | `["terminate"]` | Words that stop the entire orchestrator |

### `SpychOrchestrator` Parameters

| Parameter | Default | Description |
|---|---|---|
| `entries` | *(required)* | List of `OrchestratorEntry` dicts |
| `spych_wake_kwargs` | `None` | Extra kwargs forwarded to `SpychWake` |

---

# Python — Built-in Agents

The same agents available from the CLI can be used directly from Python.

## Claude Code CLI

```python
from spych.agents import claude_code_cli

# Say "hey claude" to trigger
claude_code_cli()
```

## Claude Code SDK

```python
from spych.agents import claude_code_sdk

# Say "hey claude" to trigger
claude_code_sdk()
```

## Codex CLI

```python
from spych.agents import codex_cli

# Say "hey codex" to trigger
codex_cli()
```

## Gemini CLI

```python
from spych.agents import gemini_cli

# Say "hey gemini" to trigger
gemini_cli()
```

## OpenCode CLI

```python
from spych.agents import opencode_cli

# Say "hey opencode" to trigger
opencode_cli()
```

## Ollama

```python
from spych.agents import ollama

# Pull the model first: ollama pull llama3.2:latest
# Say "hey llama" to trigger
ollama(model="llama3.2:latest")
```

### Coding Agent Parameters

| Parameter | `claude_code_cli` | `claude_code_sdk` | `codex_cli` | `gemini_cli` | `opencode_cli` | Description |
|---|---|---|---|---|---|---|
| `name` | `Claude` | `Claude` | `Codex` | `Gemini` | `OpenCode` | Custom display name |
| `wake_words` | `["claude", "clod", "cloud", "clawed"]` | `["claude", "clod", "cloud", "clawed"]` | `["codex"]` | `["gemini", "google"]` | `["opencode", "open code"]` | Words that trigger the agent |
| `terminate_words` | `["terminate"]` | `["terminate"]` | `["terminate"]` | `["terminate"]` | `["terminate"]` | Words that stop the listener |
| `model` | — | — | — | — | `None` | Model in `provider/model` format |
| `listen_duration` | `0` | `0` | `0` | `0` | `0` | Seconds to listen (0 = VAD auto) |
| `continue_conversation` | `True` | `True` | `True` | `True` | `True` | Resume the most recent session |
| `setting_sources` | — | `["user", "project", "local"]` | — | — | — | Claude Code settings sources |
| `show_tool_events` | `True` | `True` | `True` | `True` | `True` | Print live tool start/end events |
| `use_speaker` | `False` | `False` | `False` | `False` | `False` | Speak responses aloud via TTS |
| `speaker_voice` | `"af_heart"` | `"af_heart"` | `"af_heart"` | `"af_heart"` | `"af_heart"` | Voice name for TTS |
| `response_style` | `""` | `""` | `""` | `""` | `""` | Style preset or custom instruction |
| `allow_intermediate_responses` | `True` | `True` | `True` | `True` | `True` | Enable intermediate response chaining |
| `spych_kwargs` | — | — | — | — | — | Extra kwargs passed to `Spych` |
| `spych_wake_kwargs` | — | — | — | — | — | Extra kwargs passed to `SpychWake` |

### Ollama Parameters

| Parameter | Default | Description |
|---|---|---|
| `name` | `"Ollama"` | Custom display name |
| `wake_words` | `["llama", "ollama", "lama"]` | Words that trigger the agent |
| `terminate_words` | `["terminate"]` | Words that stop the listener |
| `model` | `"llama3.2:latest"` | Ollama model name |
| `listen_duration` | `0` | Seconds to listen (0 = VAD auto) |
| `history_length` | `10` | Past interactions to include in context |
| `host` | `"http://localhost:11434"` | Ollama instance URL |
| `use_speaker` | `False` | Speak responses aloud via TTS |
| `speaker_voice` | `"af_heart"` | Voice name for TTS |
| `response_style` | `""` | Style preset or custom instruction |
| `allow_intermediate_responses` | `True` | Enable intermediate response chaining |
| `spych_kwargs` | `None` | Extra kwargs passed to `Spych` |
| `spych_wake_kwargs` | `None` | Extra kwargs passed to `SpychWake` |

---

# Python: Building Your Own Agent

Subclass `BaseResponder`, implement `respond`, and Spych handles the rest: wake word detection, transcription, spinner UI, timing, TTS, error handling.

`respond()` must return an `AgentResponse`. Use `self.format_prompt()` to inject the JSON schema into your prompt and `self.parse_output()` to parse the result:

```python
from spych.responders import BaseResponder, AgentResponse

class MyResponder(BaseResponder):
    def respond(self, user_input: str) -> AgentResponse:
        raw = call_my_llm(self.format_prompt(user_input))
        return self.parse_output(raw)
```

A complete working example with a custom wake word:

```python
from spych import Spych, SpychOrchestrator
from spych.responders import BaseResponder, AgentResponse

class EchoResponder(BaseResponder):
    def respond(self, user_input: str) -> AgentResponse:
        return AgentResponse(
            response=f"'{self.name}' heard: {user_input}",
            summary=f"Heard: {user_input}",
            requires_user_feedback=False,
        )

SpychOrchestrator(
    entries=[
        {
            "responder": EchoResponder(
                spych_object=Spych(whisper_model="base.en"),
                listen_duration=5,
                name="TestResponder",
            ),
            "wake_words": ["test"],
            "terminate_words": ["terminate"],
        }
    ]
).start()
```

You can also subclass a built-in agent. For example, a translation agent that routes to Ollama:

```python
from spych import Spych, SpychOrchestrator
from spych.agents import OllamaResponder
from spych.responders import AgentResponse

class Spanish(OllamaResponder):
    def respond(self, user_input: str) -> AgentResponse:
        user_input = f"Translate the following to Spanish and return only the translated text: '{user_input}'"
        return super().respond(user_input)

class German(OllamaResponder):
    def respond(self, user_input: str) -> AgentResponse:
        user_input = f"Translate the following to German and return only the translated text: '{user_input}'"
        return super().respond(user_input)

spych_object = Spych(whisper_model="base.en")

SpychOrchestrator(
    entries=[
        {
            "responder": Spanish(spych_object=spych_object, name="SpanishTranslator", model="llama3.2:latest"),
            "wake_words": ["spanish"],
            "terminate_words": ["terminate"],
        },
        {
            "responder": German(spych_object=spych_object, name="GermanTranslator", model="llama3.2:latest"),
            "wake_words": ["german"],
            "terminate_words": ["terminate"],
        },
    ]
).start()
```

Think your agent would be useful to others? Open a PR or file a feature request via a [GitHub issue](https://github.com/connor-makowski/spych/issues).

---

# Python: Lower-Level API

Need more control? Use `Spych` and `SpychWake` directly.

## Transcription

```python
from spych import Spych

spych = Spych(
    whisper_model="base.en",  # tiny, small, medium, large — all faster-whisper models work
    whisper_device="cpu",     # use "cuda" for Nvidia GPU
)

print(spych.listen(duration=5))
```

See: https://connor-makowski.github.io/spych/spych/core.html

## Wake Word Detection

```python
from spych import SpychWake, Spych

spych = Spych(whisper_model="base.en", whisper_device="cpu")

def on_wake():
    print("Wake word detected! Listening...")
    print(spych.listen(duration=5))

SpychWake(
    wake_word_map={"speech": on_wake},
    whisper_model="tiny.en",
    whisper_device="cpu",
).start()
```

See: https://connor-makowski.github.io/spych/spych/wake.html

---

# API Reference

Full docs including all parameters and methods: https://connor-makowski.github.io/spych/spych.html

---

# Support

Found a bug or want a new feature? [Open an issue on GitHub](https://github.com/connor-makowski/spych/issues).

---

# Contributing

Contributions are welcome!

1. Fork the repo and clone it locally.
2. Make your changes.
3. Run tests and make sure they pass.
4. Commit atomically with clear messages.
5. Submit a pull request.

**Virtual environment setup:**
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
./utils/test.sh
```
"""

from .core import Spych
from .wake import SpychWake
from .orchestrator import SpychOrchestrator
