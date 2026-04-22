# Spych
[![PyPI version](https://badge.fury.io/py/spych.svg)](https://badge.fury.io/py/spych)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI Downloads](https://img.shields.io/pypi/dm/spych.svg?label=PyPI%20downloads)](https://pypi.org/project/spych/)

**Spych** (pronounced "speech"): talk to your computer like it's your personal assistant — and have it talk back — without sending your voice to the cloud.

A lightweight, fully offline Python toolkit for wake word detection, audio transcription, spoken AI responses, and AI integrations. Built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [PvRecorder](https://github.com/Picovoice/pvrecorder), and [Kokoro](https://github.com/hexgrad/kokoro).

- **Can Run Fully offline**: no API keys, no cloud calls, no eavesdropping (after initial setup - voice and model downloads require internet, but are cached locally for offline use thereafter)
- **Multi-threaded wake word detection**: overlapping listener windows so you rarely miss a trigger
- **Multiple wake words**: map different words to different actions in one listener
- **Spoken responses**: neural text-to-speech via [Chatterbox Turbo](https://github.com/resemble-ai/chatterbox) (high quality, zero-shot voice cloning) or [Kokoro](https://github.com/hexgrad/kokoro) (lightweight); agents speak their summaries aloud
- **Automatic follow-up listening**: when a response ends with a question, Spych listens for your reply automatically — no wake word needed
- **Live transcription**: continuous VAD-gated transcription to `.txt` and/or `.srt` files
- **Built-in agents**: for [Ollama](https://ollama.com), [Claude Code](https://docs.anthropic.com/en/docs/claude-code), [Codex](https://github.com/openai/codex), [Gemini CLI](https://github.com/google-gemini/gemini-cli), and [OpenCode](https://opencode.ai)
- **Multi-agent orchestration**: run several agents simultaneously under a single listener, each with its own wake words
- **Personalities**: named presets that bundle wake words, voice, name, and response style — e.g. `--personality jarvis`
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
spych claude
```

All agents and their parameters are supported as flags:

```bash
spych ollama --model llama3.2:latest
spych claude --setting-sources user project local
spych codex --listen-duration 8
spych opencode --model anthropic/claude-sonnet-4-5
spych gemini --wake-words gemini "hey gemini"
```

Enable spoken responses with `--use-speaker`. The TTS backend (Chatterbox Turbo or Kokoro) is downloaded on first use and cached locally — all subsequent runs are fully offline:

```bash
spych claude --use-speaker
spych ollama --model llama3.2:latest --use-speaker --speaker-voice bm_george
spych claude --use-speaker --response-style concise
```

Personality presets bundle wake words, voice, name, and response style into a single flag:

```bash
spych claude --personality jarvis
spych ollama --model llama3.2:latest --personality jarvis
```

A global `--theme` flag controls the terminal colour output and must be placed before the agent name:

```bash
spych --theme light claude
spych --theme solarized ollama --model llama3.2:latest
```

Available themes: `dark` (default), `light`, `solarized`, `mono`.

Live transcription is also available via the CLI:

```bash
spych live
spych live --output-path meeting --output-format srt
spych live --terminate-words "stop recording"
spych live --no-timestamps --whisper-model small.en
```

Multiple agents can be run by creating one terminal session per agent and setting `--wake-words` to be different per agent. In this way you can create 3 claude agents with different wake words.

- A Multi agent mode is also available via the CLI, but has some limitations.
- See the "Multi-agent" section below for more details.

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
| `name` | `Claude` | `Claude` | `Codex` | `Gemini` | `OpenCode` | Custom display name for the agent |
| `wake_words` | `["claude", "clod", "cloud", "clawed"]` | `["claude", "clod", "cloud", "clawed"]` | `["codex"]` | `["gemini", "google"]` | `["opencode", "open code"]` | Words that trigger the agent |
| `terminate_words` | `["terminate"]` | `["terminate"]` | `["terminate"]` | `["terminate"]` | `["terminate"]` | Words that stop the listener |
| `model` | - | - | - | - | `None` | Model in `provider/model` format |
| `listen_duration` | `0` | `0` | `0` | `0` | `0` | Seconds to listen after wake word (0 = VAD auto) |
| `continue_conversation` | `True` | `True` | `True` | `True` | `True` | Resume the most recent session |
| `setting_sources` | - | `["user", "project", "local"]` | - | - | - | Claude Code local settings to load |
| `show_tool_events` | `True` | `True` | `True` | `True` | `True` | Print live tool start/end events |
| `use_speaker` | `False` | `False` | `False` | `False` | `False` | Speak responses aloud via TTS |
| `speaker_voice` | `"af_heart"` | `"af_heart"` | `"af_heart"` | `"af_heart"` | `"af_heart"` | Voice name for spoken responses |
| `response_style` | `""` | `""` | `""` | `""` | `""` | Style preset or custom instruction for spoken output |
| `spych_kwargs` | - | - | - | - | - | Extra kwargs passed to `Spych` |
| `spych_wake_kwargs` | - | - | - | - | - | Extra kwargs passed to `SpychWake` |

### Ollama Parameters

| Parameter | Default | Description |
|---|---|---|
| `name` | `"Ollama"` | Custom display name for the agent |
| `wake_words` | `["llama", "ollama", "lama"]` | Words that trigger the agent |
| `terminate_words` | `["terminate"]` | Words that stop the listener |
| `model` | `"llama3.2:latest"` | Ollama model name |
| `listen_duration` | `0` | Seconds to listen after wake word (0 = VAD auto) |
| `history_length` | `10` | Past interactions to include in context |
| `host` | `"http://localhost:11434"` | Ollama instance URL |
| `use_speaker` | `False` | Speak responses aloud via TTS |
| `speaker_voice` | `"af_heart"` | Voice name for spoken responses |
| `response_style` | `""` | Style preset or custom instruction for spoken output |
| `spych_kwargs` | `None` | Extra kwargs passed to `Spych` |
| `spych_wake_kwargs` | `None` | Extra kwargs passed to `SpychWake` |

---

# Summaries & Text-to-Speech (TTS)

Every agent response includes both a full `response` (printed to the terminal) and a short `summary`. The summary is always printed below long responses (over ~200 characters) so you can quickly scan what was said without scrolling. It is written to be clean prose with no file paths or special characters.

Any agent can also speak the summary aloud using the built-in neural TTS engine. 

### TTS Backends & Fallback

Spych uses a tiered fallback system for TTS to balance quality and performance:

1.  **Chatterbox** (High Quality / Priority): Best for natural sounding voices and zero-shot cloning. Slower and requires more resources. **Required for Python 3.14+.**
2.  **Kokoro** (Lightweight): Very fast and efficient. Ideal for edge devices (like a Raspberry Pi). *Note: Not supported on Python 3.14+.*

### Installation Recommendations

We recommend installing with **Kokoro** for most users (Python <= 3.13) as it is significantly faster and uses fewer resources.

Choose **Chatterbox** if:
- You need high-quality voice cloning (zero-shot)
- You want to use custom voice samples (`.wav` files)
- You are running on **Python 3.14+**

#### Install and Run with your preferred TTS engine:

Note: If you are using python 3.14+, you will automatically install chatterbox on `pip install spych`

Note: If you are using python 3.13-, you will automatically install kokoro on `pip install spych`

By default, you will use chatterbox first if it is installed, otherwise, you will use kokoro if it is installed.

```bash
# Recommended for most users (Fast, lightweight)
pipx install "spych[kokoro]"

# For high-quality voice cloning or Python 3.14+
pipx install "spych[chatterbox]"

```

Enable TTS with `--use-speaker` (CLI) or `use_speaker=True` (Python). You can explicitly choose a backend with `--speaker-backend`:

```bash
spych claude --use-speaker --speaker-backend kokoro
spych ollama --use-speaker --speaker-backend chatterbox
```

When TTS is active, short responses are spoken verbatim; longer ones use the `summary`. If the spoken response requires user feedback, Spych automatically listens for a follow-up answer — no wake word required.

### One-Shot Voice Cloning (Personalization)

One-shot cloning allows you to create a digital twin of any voice from a short audio sample. This feature is powered by **Chatterbox Turbo** and is not supported by the lightweight Kokoro backend.

#### 1. Record your profile
Run the following command to record a 10-second sample of your voice. Spych will prompt you with a specific passage to read:

```bash
spych profile_my_voice --name my_voice
```

#### 2. Use your custom voice
Once recorded, your voice profile is saved to the local cache and can be used by any agent. You must specify the **Chatterbox** backend to use custom voices:

```bash
spych claude --use-speaker --speaker-voice my_voice --speaker-backend chatterbox
```

### Using an alternative custom voice
You can also use any `.wav` file as a voice profile with Chatterbox.

Simply specify the path to your `.wav` file instead of a profile name:

```bash
spych claude --use-speaker --speaker-voice /path/to/my_voice.wav --speaker-backend chatterbox
```

*Note: Custom `.wav` profiles are only compatible with the Chatterbox backend. If you attempt to use a custom voice with Kokoro, it will fall back to using chatterbox if installed. If chatterbox is not installed, it will fall back to the default voice for kokoro.* 

### Available Voices

The same voice names (e.g. `af_heart`, `bm_george`) work for both backends. Chatterbox uses `.wav` reference files for zero-shot cloning; Kokoro uses `.pt` voice tensors. Voice files are downloaded automatically on first use.

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

---

# Personalities

Personalities are named presets that bundle a wake word list, voice, display name, and response style into a single flag. They are applied as defaults — any explicit flag overrides the preset.

```bash
spych claude --personality jarvis
# equivalent to:
spych claude --name "J.A.R.V.I.S." --wake-words jarvis jarves \\
             --speaker-voice bm_george --use-speaker \\
             --response-style jarvis
```

### Available Personalities

| Name | Wake words | Voice | Style |
|---|---|---|---|
| `jarvis` | `jarvis`, `jarves` | `bm_george` | `jarvis` — precise, dry wit, "sir" |
| `pirate` | `blackbeard`, `pirate`, `ahoy` | `am_fenrir` | `pirate` — pirate speak, colorful |
| `news_anchor` | `bella`, `news anchor`, `anchor` | `af_bella` | `news_anchor` — professional broadcast tone |
| `robot` | `rob`, `robot` | `am_michael` | `robot` — monotone, literal |
| `caveman` | `er`, `ur`, `caveman`, `cave man` | `am_puck` | `caveman` — very simple, direct |

### Response Styles

The `response_style` parameter shapes how the LLM formats its output. Named presets:

| Style | Description |
|---|---|
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
| `jarvis` | J.A.R.V.I.S. from Iron Man — precise, dry wit, addresses user as "sir" |

You can also pass any custom instruction string directly: `response_style="Reply in exactly one sentence."`.

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
| `no_speech_threshold` | `0.4` | Whisper segments with `no_speech_prob` above this are discarded |
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
spych multi --agents claude gemini

# Include Ollama with a specific model
spych multi --agents claude ollama --ollama-model llama3.2:latest

# Tune listen duration across all agents
spych multi --agents claude codex --listen-duration 8
```

### Multi-agent CLI Parameters

| Flag | Default | Description |
|---|---|---|
| `--agents` | *(required)* | One or more agent names to run: `claude` (`claude_code_cli`), `claude_sdk` (`claude_code_sdk`), `codex` (`codex_cli`), `gemini` (`gemini_cli`), `opencode` (`opencode_cli`), `ollama` |
| `--terminate-words` | `["terminate"]` | Words that stop all agents |
| `--listen-duration` | `5` | Seconds to listen after a wake word |
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

Not using any of the above? No problem. Subclass `BaseResponder`, implement `respond`, and you're done. Spych handles the rest: listening, transcription, spinner UI, timing, TTS, error handling, all of it.

`respond()` must return an `AgentResponse`. Use `self.format_prompt()` to inject the JSON schema into your outgoing prompt and `self.parse_output()` to parse the result:

```python
from spych.responders import BaseResponder, AgentResponse

class MyResponder(BaseResponder):
    def respond(self, user_input: str) -> AgentResponse:
        raw = call_my_llm(self.format_prompt(user_input))
        return self.parse_output(raw)
```

If you just want to echo input (e.g. for testing), construct `AgentResponse` directly:

```python
from spych.responders import BaseResponder, AgentResponse

class EchoResponder(BaseResponder):
    def respond(self, user_input: str) -> AgentResponse:
        return AgentResponse(
            response=f"'{self.name}' heard: {user_input}",
            summary=f"Heard: {user_input}",
            requires_user_feedback=False,
        )
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

The orchestrator can also handle multiple custom agents at once, each with their own wake words. For example, you can make a translation agent that listens for "Spanish" or "German" and routes to the appropriate responder:

> Note: To run this example, you will need to have Ollama running and an Ollama model that can do translations. You can use `llama3.2:latest` or any other model you have set up for this purpose.

```python
from spych import Spych, SpychOrchestrator
from spych.agents import OllamaResponder
from spych.responders import AgentResponse

class Spanish(OllamaResponder):
    def respond(self, user_input: str) -> AgentResponse:
        user_input = f"Translate the following text to Spanish and return only the translated text: '{user_input}'"
        return super().respond(user_input)

class German(OllamaResponder):
    def respond(self, user_input: str) -> AgentResponse:
        user_input = f"Translate the following text to German and return only the translated text: '{user_input}'"
        return super().respond(user_input)

SpychOrchestrator(
    entries=[
        {
            "responder": Spanish(
                spych_object=Spych(whisper_model="base.en"),
                name="SpanishTranslator",
                model="llama3.2:latest",
            ),
            "wake_words": ["spanish"],
            "terminate_words": ["terminate"],
        },
        {
            "responder": German(
                spych_object=Spych(whisper_model="base.en"),
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