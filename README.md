# Spych
[![PyPI version](https://badge.fury.io/py/spych.svg)](https://badge.fury.io/py/spych)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI Downloads](https://img.shields.io/pypi/dm/spych.svg?label=PyPI%20downloads)](https://pypi.org/project/spych/)

**Spych** (pronounced "speech"): talk to your computer like its your personal assistant without sending your voice to the cloud.

A lightweight, fully offline Python toolkit for wake word detection, audio transcription, and AI integrations. Built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and [PvRecorder](https://github.com/Picovoice/pvrecorder).

- **Fully offline**: no API keys, no cloud calls, no eavesdropping
- **Multi-threaded wake word detection**: overlapping listener windows so you rarely miss a trigger
- **Multiple wake words**: map different words to different actions in one listener
- **Live transcription**: continuous VAD-gated transcription to `.txt` and/or `.srt` files
- **Built-in agents**: for [Ollama](https://ollama.com), [Claude Code](https://docs.anthropic.com/en/docs/claude-code), [Codex](https://github.com/openai/codex), [Gemini CLI](https://github.com/google-gemini/gemini-cli), and [OpenCode](https://opencode.ai)
- **Multi-agent orchestration**: run several agents simultaneously under a single listener, each with its own wake words
- **Extensible**: subclass `BaseResponder` to build your own agents with custom wake words and logic

**API Docs**: https://connor-makowski.github.io/spych/spych.html


# Setup

## Installation

### Recommended: pipx (strongly recommended)

Install Spych globally using [pipx](https://pipx.pypa.io/stable/installation/):

```bash
pipx install spych
```

### Alternative: pip

Install using pip (requires Python 3.11+):

```bash
pip install spych
```

---

# CLI

Once installed, `spych` is available as a command anywhere on your machine. You will still need to set up your respective agents before using them. See the docs below for setup instructions. Navigate to your project directory and launch any agent directly:

```bash
cd ~/my_project
spych claude_code_cli
```

All agents and their parameters are supported as flags:

```bash
spych ollama --model llama3.2:latest
spych claude_code_sdk --setting-sources user project local
spych codex_cli --listen-duration 8
spych opencode_cli --model anthropic/claude-sonnet-4-5
spych gemini_cli --wake-words gemini "hey gemini"
```

Live transcription is also available via the CLI:

```bash
spych live
spych live --output-path meeting --output-format srt
spych live --terminate-words "stop recording"
spych live --no-timestamps --whisper-model small.en
```

Multi-agent mode is also available via the CLI. See the "Multi-agent" section below for more details.

```bash
spych multi --agents claude_code_sdk ollama --ollama-model llama3.2:latest --listen-duration 8
```

Run `spych --help` or `spych <agent> --help` to see all available options.

---

# Quick Start: Voice Agents

The fastest path from zero to voice-controlled AI. These one-liners handle everything: wake word detection, transcription, and routing your speech to the target agent.

## Ollama

Talk to a local LLM entirely offline. Requires [Ollama](https://ollama.com) installed and running.

For this example, we'll use the free `llama3.2:latest` model, but any Ollama model will work. For this example run: `ollama pull llama3.2:latest`.
```python
from spych.agents import ollama

# Pull the model first: ollama pull llama3.2:latest
# Then say "hey llama" to trigger
ollama(model="llama3.2:latest")
```

## Claude Code CLI

Voice-control Claude Code directly from your terminal. Requires [Claude Code](https://docs.anthropic.com/en/docs/claude-code) installed and authenticated. See: https://code.claude.com/docs/en/quickstart. Make sure you can run `claude code` commands in your terminal before trying this. 

Note: This can pull from your `.claude` folder in your user directory or from the project directory, so you can have different settings for different projects if you like.


```python
from spych.agents import claude_code_cli

# Say "hey claude" to trigger
claude_code_cli()
```

## Claude Code SDK

Same as above but uses the Claude Agent SDK via a subprocess worker instead of the CLI. This is great for a lightweight setup with better tool call feedback loops, but you will still need to be authenticated with the SDK and have your tools set up. See: https://platform.claude.com/docs/en/agent-sdk/overview for setup instructions. 

Note: This can pull from your `.claude` folder in your user directory or from the project directory, so you can have different settings for different projects if you like.

```python
from spych.agents import claude_code_sdk

# Say "hey claude" to trigger
claude_code_sdk()
```

## Codex CLI

Voice-control OpenAI's Codex agent. Requires [Codex CLI](https://github.com/openai/codex) installed and authenticated. Make sure you can run `codex` commands in your terminal before trying this.

```python
from spych.agents import codex_cli

# Say "hey codex" to trigger
codex_cli()
```

## Gemini CLI

Voice-control Google's Gemini agent. Requires [Gemini CLI](https://github.com/google-gemini/gemini-cli) installed and authenticated. Make sure you can run `gemini` commands in your terminal before trying this.

```python
from spych.agents import gemini_cli

# Say "hey gemini" to trigger
gemini_cli()
```

## OpenCode CLI

Voice-control the OpenCode agent. Requires [OpenCode](https://opencode.ai) installed and authenticated. Make sure you can run `opencode` commands in your terminal before trying this.

```python
from spych.agents import opencode_cli

# Say "hey opencode" to trigger
opencode_cli()
```

> 💡 **Pro tip:** Saying "Hey Llama" or "Hey Claude" tends to trigger more reliably than just the bare wake word.

All agents accept a `terminate_words` list (default: `["terminate"]`). Say the word or use `ctrl+c` to stop the listener cleanly.

### Coding Agent Parameters

| Parameter | `claude_code_cli` | `claude_code_sdk` | `codex_cli` | `gemini_cli` | `opencode_cli` | Description |
|---|---|---|---|---|---|---|
| `wake_words` | `["claude", "clod", "cloud", "clawed"]` | `["claude", "clod", "cloud", "clawed"]` | `["codex"]` | `["gemini"]` | `["opencode", "open code"]` | Words that trigger the agent |
| `terminate_words` | `["terminate"]` | `["terminate"]` | `["terminate"]` | `["terminate"]` | `["terminate"]` | Words that stop the listener |
| `model` | - | - | - | - | `None` | Model in `provider/model` format |
| `listen_duration` | `0` | `0` | `0` | `0` | `0` | Seconds to listen after wake word (0 = VAD auto) |
| `continue_conversation` | `True` | `True` | `True` | `True` | `True` | Resume the most recent session |
| `setting_sources` | - | `["user", "project", "local"]` | - | - | - | Claude Code local settings to load |
| `show_tool_events` | `True` | `True` | `True` | `True` | `True` | Print live tool start/end events |
| `spych_kwargs` | - | - | - | - | - | Extra kwargs passed to `Spych` |
| `spych_wake_kwargs` | - | - | - | - | - | Extra kwargs passed to `SpychWake` |

### Ollama Parameters

| Parameter | Default | Description |
|---|---|---|
| `wake_words` | `["llama", "ollama", "lama"]` | Words that trigger the agent |
| `terminate_words` | `["terminate"]` | Words that stop the listener |
| `model` | `"llama3.2:latest"` | Ollama model name |
| `listen_duration` | `0` | Seconds to listen after wake word (0 = VAD auto) |
| `history_length` | `10` | Past interactions to include in context |
| `host` | `"http://localhost:11434"` | Ollama instance URL |
| `spych_kwargs` | `None` | Extra kwargs passed to `Spych` |
| `spych_wake_kwargs` | `None` | Extra kwargs passed to `SpychWake` |

---

# Live Transcription

`SpychLive` continuously records from the microphone using VAD and writes the transcript to disk in real time. No wake word required — it transcribes everything until stopped.

## Python

```python
from spych.live import SpychLive

live = SpychLive(
    output_format="srt",         # "txt", "srt", or "both"
    output_path="my_transcript", # written to my_transcript.srt
    show_timestamps=True,
    stop_key="q",                # type q + Enter to stop
    terminate_words=["stop recording"],
)
live.start()
```

## CLI

```bash
spych live                                           # writes transcript.srt
spych live --output-path meeting --output-format both
spych live --terminate-words "stop recording"
spych live --no-timestamps --whisper-model small.en
```

### `SpychLive` Parameters

| Parameter | Default | Description |
|---|---|---|
| `output_format` | `"srt"` | Output format(s): `"txt"`, `"srt"`, or `"both"` |
| `output_path` | `"transcript"` | Base path without extension; extensions are appended automatically |
| `show_timestamps` | `True` | Prepend `[HH:MM:SS]` timestamps to terminal and `.txt` output |
| `stop_key` | `"q"` | Key (then Enter) to stop the session |
| `terminate_words` | `None` | Spoken words that stop the session (detected after transcription, ~1–3s latency) |
| `on_terminate` | `None` | No-argument callback executed when a terminate word fires |
| `device_index` | `-1` | Microphone device index; `-1` uses system default |
| `whisper_model` | `"base.en"` | faster-whisper model name |
| `whisper_device` | `"cpu"` | Device for inference: `"cpu"` or `"cuda"` |
| `whisper_compute_type` | `"int8"` | Compute precision: `"int8"`, `"float16"`, or `"float32"` |
| `no_speech_threshold` | `0.3` | Whisper segments with `no_speech_prob` above this are discarded |
| `speech_threshold` | `0.5` | Silero VAD probability above which a frame is considered speech onset |
| `silence_threshold` | `0.35` | Silero VAD probability below which a frame is considered silence during speech |
| `silence_frames_threshold` | `20` | Consecutive silent frames (~32ms each) required to close a segment (~640ms) |
| `speech_pad_frames` | `5` | Pre-roll frame count and onset confirmation threshold (~160ms) |
| `max_speech_duration_s` | `30.0` | Hard cap on a single segment in seconds |
| `context_words` | `32` | Trailing transcript words passed as `initial_prompt` for contextual accuracy |

---

# Multi-agent

Run several agents simultaneously under a single listener, each bound to its own wake words. Say "hey claude" to talk to Claude, "hey llama" to talk to Ollama — all in the same terminal session.

## CLI

```bash
# Two agents, default wake words
spych multi --agents claude_code_cli gemini_cli

# Include Ollama with a specific model
spych multi --agents claude_code_cli ollama --ollama-model llama3.2:latest

# Tune listen duration across all agents
spych multi --agents claude_code_sdk codex_cli --listen-duration 8
```

### Multi-agent CLI Parameters

| Flag | Default | Description |
|---|---|---|
| `--agents` | *(required)* | One or more agent names to run: `claude_code_cli`, `claude_code_sdk`, `codex_cli`, `gemini_cli`, `opencode_cli`, `ollama` |
| `--terminate-words` | `["terminate"]` | Words that stop all agents |
| `--listen-duration` | `0` | Seconds to listen after a wake word (0 = VAD auto) |
| `--continue-conversation` | `true` | Resume the most recent session for each coding agent |
| `--show-tool-events` | `true` | Print live tool start/end events |
| `--ollama-model` | `llama3.2:latest` | Ollama model. Only used when `ollama` is in `--agents` |
| `--ollama-host` | `http://localhost:11434` | Ollama instance URL. Only used when `ollama` is in `--agents` |
| `--ollama-history-length` | `10` | Ollama context history length. Only used when `ollama` is in `--agents` |
| `--opencode-model` | `None` | OpenCode model in `provider/model` format. Only used when `opencode_cli` is in `--agents` |
| `--setting-sources` | `["user", "project", "local"]` | Claude Code SDK setting sources. Only used when `claude_code_sdk` is in `--agents` |

## Python

Use `SpychOrchestrator` directly to mix any combination of responders with custom wake words.

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
| `responder` | ✓ | - | A `BaseResponder` instance |
| `wake_words` | ✓ | - | Words that trigger this responder. Must be unique across all entries |
| `terminate_words` | | `["terminate"]` | Words that stop the entire orchestrator. Merged across all entries |

### `SpychOrchestrator` Parameters

| Parameter | Default | Description |
|---|---|---|
| `entries` | *(required)* | List of `OrchestratorEntry` dicts — see table above |
| `spych_wake_kwargs` | `None` | Extra kwargs forwarded to `SpychWake` (e.g. `whisper_model`, `wake_listener_count`) |

---

# Building Your Own Agent

Not using any of the above? No problem. Subclass `BaseResponder`, implement `respond`, and you're done. Spych handles the rest: listening, transcription, spinner UI, timing, error handling, all of it.
```python
from spych.responders import BaseResponder

class MyResponder(BaseResponder):
    def respond(self, user_input: str) -> str:
        return f"'{self.name}' heard: {user_input}"
```

A complete working example with a custom wake word:
```python
from spych import Spych,SpychOrchestrator
from spych.responders import BaseResponder

class MyResponder(BaseResponder):
    def respond(self, user_input: str) -> str:
        return f"'{self.name}' heard: {user_input}"

SpychOrchestrator(
    entries=[
        {
            "responder": MyResponder(
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

The orchestrator can also handle multiple custom agents at once, each with their own wake words. For example, you can make a translation agent that listens for "Spanish" or "German" and routes to the appropriate responder:

> Note: To run this example, you will need to have Ollama running and an Ollama model that can do translations. You can use `llama3.2:latest` or any other model you have set up for this purpose.

```python
from spych import Spych,SpychOrchestrator
from spych.agents import OllamaResponder

class Spanish(OllamaResponder):
    def respond(self, user_input: str) -> str:
        user_input = f"Translate the following text to Spanish and return only the translated text: '{user_input}'"
        response = super().respond(user_input)
        return response
    
class German(OllamaResponder):
    def respond(self, user_input: str) -> str:
        user_input = f"Translate the following text to German and return only the translated text: '{user_input}'"
        response = super().respond(user_input)
        return response

SpychOrchestrator(
    entries=[
        {
            "responder": Spanish(
                spych_object=Spych(whisper_model="base.en"),
                listen_duration=5,
                name="SpanishTranslator",
                model="llama3.2:latest",
            ),
            "wake_words": ["spanish"],
            "terminate_words": ["terminate"],
        },
        {
            "responder": German(
                spych_object=Spych(whisper_model="base.en"),
                listen_duration=5,
                name="GermanTranslator",
                model="llama3.2:latest",
            ),
            "wake_words": ["german"],
            "terminate_words": ["terminate"],
        }
    ]
).start()
```

## Custom Agent Contributions

Think your agent would be useful to others? Open a PR or file a feature request via a GitHub issue. Contributions are very welcome.

---

# Lower-Level API

Need more control? Use `SpychWake` and `Spych` directly.

## Listen and Transcribe

`Spych` records from the mic and returns a transcription string.
```python
from spych import Spych

spych = Spych(
    whisper_model="base.en",  # or tiny, small, medium, large -> all faster-whisper models work
    whisper_device="cpu",     # use "cuda" if you have an Nvidia GPU
)

print(spych.listen(duration=5))
```

See: https://connor-makowski.github.io/spych/spych/core.html

## Wake Word Detection

`SpychWake` runs multiple overlapping listener threads and fires a callback when a wake word is detected.
```python
from spych import SpychWake, Spych

spych = Spych(whisper_model="base.en", whisper_device="cpu")

def on_wake():
    print("Wake word detected! Listening...")
    print(spych.listen(duration=5))

wake = SpychWake(
    wake_word_map={"speech": on_wake},
    whisper_model="tiny.en",
    whisper_device="cpu",
)

wake.start()
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