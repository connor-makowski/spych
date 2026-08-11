# spych: Development Guide

## Project Purpose

`spych` (pronounced "speech") is a lightweight, fully offline Python toolkit for voice-driven AI interactions. Core capabilities:

- **Wake word detection** — multi-threaded overlapping listener windows for continuous, low-latency detection
- **Voice transcription** — faster-whisper with optional Silero VAD gating for hands-free, auto-terminated recording
- **AI agent responders** — built-in support for Ollama, Claude Code (CLI and SDK), Antigravity (agy), Codex, and OpenCode
- **Multi-agent orchestration** — bind multiple responders to different wake words under a single listener session
- **Live transcription** — continuous VAD-gated transcription to `.txt` and/or `.srt` files
- **Live translation** — continuous VAD-gated transcription + Ollama-powered translation with bilingual output
- **Extensible** — subclass `BaseResponder`, implement one method, and plug into the orchestration system

All core processing (VAD, transcription, wake word detection) runs fully offline on the local machine.

---

## Directory Layout (relevant```
.agents/skills/            # Agent skill definitions (add-responder, add-test, test, prettify, debug, etc.)
spych/
  __init__.py              # Public exports: Spych, SpychWake, SpychOrchestrator, BaseResponder,
                           #   AgentResponse, Speaker, PERSONALITIES, get_personality, get_response_style
  core.py                  # Spych — faster-whisper transcription engine with Silero VAD
  wake.py                  # SpychWake + SpychWakeListener — multi-threaded wake word detection
  responders.py            # BaseResponder, AgentResponse — abstract base + structured response dataclass
  orchestrator.py          # SpychOrchestrator — multi-agent coordinator with shared spinner
  live.py                  # SpychLive — continuous VAD-gated transcription to disk
  live_translation.py      # SpychLiveTranslation — VAD-gated transcription + Ollama translation with bilingual output
  speaker/
    __init__.py            # Re-exports Speaker
    speaker.py             # Speaker — Chatterbox Turbo (primary) or Kokoro (fallback) TTS
    backends.py            # ChatterboxBackend, KokoroBackend, ChatterboxMultilingualBackend, get_backend
    chatterbox.py          # SpychChatterboxTTS — standalone Chatterbox Turbo implementation
    chatterbox_multilingual.py  # SpychChatterboxMultilingualTTS — multilingual Chatterbox implementation
  cli.py                   # CLI entry point (spych subcommands)
  cli_tools.py             # Theme, CliSpinner, CliPrinter — terminal UI utilities
  dashboard.py             # AgentDashboard — rich TUI for live agent interaction
  spinners.py              # Spinner frame definitions (BRAILLE, ARC, MOON, etc.)
  utils.py                 # Recorder, Notify, get_response_style, PERSONALITIES, get_personality
  agents/
    __init__.py            # Exports all agent factories and responder classes
    claude.py              # LocalClaudeCodeCLIResponder + LocalClaudeCodeSDKResponder + factories
    ollama.py              # OllamaResponder + ollama() factory
    agy.py                 # LocalAntigravityCLIResponder + antigravity_cli() factory
    codex.py               # LocalCodexCLIResponder + codex_cli() factory
    opencode.py            # LocalOpenCodeCLIResponder + opencode_cli() factory
    sdk_workers/
      claude_sdk_worker.py # Subprocess worker for Claude Agent SDK communication
test/
  NN_*.py                  # 14 test scripts; run via pytest or nox
utils/
  prettify.py              # autoflake (unused imports) + black
  docs.py                  # Generate pdoc HTML docs — do NOT run unless releasing
noxfile.py                 # nox sessions: pytest across Python versions, chatterbox & kokoro testing
pyproject.toml             # Package metadata, pytest + black config, dependencies
skills/                    # Reference skills (preserved for structure comparison)
```

---

## Development Commands

All development tasks use `uv`, `pytest`, and `nox`:

| Command | What it does |
|---|---|
| `nox` | Run standard `pytest` suite across Python 3.11–3.13 (Kokoro on <3.13, Chatterbox on 3.13+) |
| `nox -s prettify` | Format with autoflake + black via `utils/prettify.py` |
| `pytest` or `uv run pytest` | Run test suite in current virtual environment |
| `uv run python utils/prettify.py` | Format code locally using `uv` |
| `nox -s docs` | Regenerate pdoc documentation |

**Test runner** (`pytest` / `nox`): Discovers and runs all `test/NN_*.py` test scripts.

**Docs**: **DO NOT generate docs**. Docs are regenerated and versioned at release time by the maintainer only.

---

## Agent Skills (`.agents/skills/`)

The repository includes standard agent skills under `.agents/skills/` to guide development workflows:

| Skill | Purpose |
|---|---|
| `add-responder` | Guide for implementing new AI agent responders subclassing `BaseResponder`. |
| `add-test` | Blueprint for creating standalone numbered test scripts (`test/NN_description.py`). |
| `test` | Workflows and commands for executing test suites (`nox` across Python versions, `pytest`). |
| `prettify` | Guidelines and commands for code formatting (`nox -s prettify` or `python utils/prettify.py`). |
| `debug` | Diagnostic checklists for audio recording, VAD, wake words, LLM parsing, and TTS. |
| `add-personality` | Steps for adding character/voice personality presets to `PERSONALITIES`. |
| `live-pipeline` | Architecture and modification steps for continuous transcription and translation. |
| `dashboard` | Architecture and testing procedures for `AgentDashboard` rich TUI. |

---

## Core Architecture

### Key Classes

**`Spych`** (`core.py`) — transcription engine:
- Wraps `faster-whisper` for speech-to-text
- Two recording modes: fixed-duration and VAD-gated (`duration=0` or `duration="auto"`)
- VAD uses Silero with hysteresis (separate speech onset and silence offset thresholds)
- `listen(duration, device_index) -> str`: records audio and returns transcribed text

**`SpychWake`** (`wake.py`) — wake word detection:
- Spawns `wake_listener_count` overlapping `SpychWakeListener` threads, each recording for `wake_listener_time` seconds
- Threads stagger their start times to provide seamless continuous coverage
- Matches transcribed text against `wake_word_map` keys; fires the associated callback
- `start()`: blocking loop; `stop()`: signals all listeners to exit

**`AgentResponse`** (`responders.py`) — structured response dataclass:
- `response: str` — full text printed to terminal
- `summary: str` — short spoken-word-friendly version used by `Speaker`
- `requires_user_feedback: bool` — True when the response ends with a question, triggering `spoken_follow_up_loop`

**`BaseResponder`** (`responders.py`) — abstract agent base:
- Subclass and implement `respond(user_input: str) -> AgentResponse` — the only required method
- `format_prompt(prompt)` — wraps user input with JSON schema + style hint; call before sending to LLM
- `parse_output(raw_text)` — parses LLM JSON into `AgentResponse`; handles markdown fences and embedded JSON
- Handles the full voice cycle: listen → transcribe → respond → print → (optional TTS speak)
- Optional hooks: `healthcheck()`, `on_before_respond()`, `on_after_respond()`
- `__call__() -> AgentResponse | None`: runs one complete voice cycle
- `allow_intermediate_responses: bool` (default `True`) — when `False`, disables intermediate response chaining.

**`SpychOrchestrator`** (`orchestrator.py`) — multi-agent coordinator:
- Accepts a list of `OrchestratorEntry` dicts, each with `responder`, `wake_words`, and `terminate_words`
- Creates a single shared `CliSpinner` injected into all responders to prevent output conflicts
- Builds a flat `wake_word → responder.__call__` map and passes it to `SpychWake`
- `start()`: blocking; triggers the correct responder when its wake word is heard

**`SpychLive`** (`live.py`) — live transcription pipeline:
- Producer-consumer architecture with three threads: `VADRecorder` → `Transcriber` → `Writer`
- Writes to `.txt`, `.srt`, or both; maintains a context buffer (~128 words) for whisper `initial_prompt`
- Stoppable via keystroke, spoken terminate word, or `KeyboardInterrupt`

**`SpychLiveTranslation`** (`live_translation.py`) — live transcription + translation pipeline:
- Extends the live transcription pattern: `VADRecorder` → `TranslatingTranscriber` → `TranslationWriter`
- `TranslatingTranscriber` transcribes in the source language then calls Ollama (via `requests`) to translate
- `TranslationWriter` writes bilingual output — `[time](src_lang) original` + `[time](tgt_lang) translated` — and optionally speaks the translation via `Speaker`
- Whisper model `.en` suffix is stripped automatically when `input_language != "en"`
- If Ollama is unreachable, translation falls back to `[translation unavailable]` without interrupting the session

**`AgentDashboard`** (`dashboard.py`) — TUI:
- Renders a live, interactive terminal dashboard in the alternate screen buffer.
- Features real-time tool tracking, thought streaming, and conversation history.
- Includes a scrollable "All Logs" mode and optimized text wrapping with internal caching.
- Handles keyboard input for scrolling and mode toggling via a dedicated input thread.

### Built-in Agents (`spych/agents/`)

| Class | Factory | Backend |
|---|---|---|
| `OllamaResponder` | `ollama()` | Local Ollama REST API |
| `LocalClaudeCodeCLIResponder` | `claude_code_cli()` | `claude` CLI subprocess |
| `LocalClaudeCodeSDKResponder` | `claude_code_sdk()` | Claude Agent SDK subprocess worker |
| `LocalAntigravityCLIResponder` | `antigravity_cli()` / `agy_cli()` | `agy` CLI subprocess |
| `LocalCodexCLIResponder` | `codex_cli()` | `codex` CLI subprocess |
| `LocalOpenCodeCLIResponder` | `opencode_cli()` | `opencode` CLI subprocess |

Each factory function: creates a `Spych` instance, wraps it in the responder, and starts a `SpychOrchestrator`. User-supplied `spych_kwargs` and `spych_wake_kwargs` dicts override factory defaults.

### Inheritance Hierarchy

```
Notify (utils.py — base logging/notification)
├── Spych (core.py)
├── SpychWake (wake.py)
│   └── SpychWakeListener
├── BaseResponder (responders.py)
│   ├── OllamaResponder
│   ├── LocalClaudeCodeCLIResponder
│   ├── LocalClaudeCodeSDKResponder
│   ├── LocalAntigravityCLIResponder
│   ├── LocalCodexCLIResponder
│   └── LocalOpenCodeCLIResponder
└── SpychLive (live.py)
```

### BaseResponder Voice Cycle

```
__call__()
├── on_listen_start()         # Spinner: "Listening..."
├── listen()                  # VAD-gated audio → transcription
├── on_listen_end()           # Brief spinner pause
├── on_user_input()           # Print user input, restart spinner
├── on_before_respond()       # (hook — optional override)
├── respond()                 # ← subclass implements this; returns AgentResponse
│   ├── format_prompt()       #   wraps input with JSON schema + style hint
│   └── parse_output()        #   parses LLM JSON into AgentResponse
├── on_after_respond()        # (hook — optional override)
├── on_response()             # Print response + summary + elapsed time
└── if use_speaker:
    ├── speak_to_user()       # Background thread: TTS speaks summary
    └── spoken_follow_up_loop() # If requires_user_feedback, listen → respond loop
    else:
    └── wait_for_next_wake_word() # Print divider, restart spinner
```

### Structured Agent Responses

All built-in agents instruct the LLM to return a JSON object via `format_prompt()`. The LLM response is then parsed by `parse_output()` into an `AgentResponse`. This eliminates the extra LLM round-trip previously needed to generate a spoken summary.

**Implementing a custom agent:**
```python
def respond(self, user_input: str) -> AgentResponse:
    raw = call_my_llm(self.format_prompt(user_input))
    return self.parse_output(raw)
```

### Personalities

`PERSONALITIES` in `utils.py` maps preset names to agent kwargs bundles. Each entry contains `name`, `wake_words`, `speaker_voice`, `use_speaker`, and `response_style`. The `--personality` CLI flag applies these as defaults before any explicit CLI flags.

**Adding a personality:** add an entry to `PERSONALITIES` and (optionally) a matching key in the `styles` dict inside `get_response_style()`.

---

## Test Structure

Tests are in `test/`. Each file is a standalone Python script run via `pytest` or `nox`.

**Naming convention:** `NN_description.py` or `test_*.py`

**Rough groupings:**
- `01`: Wake word detection + listener setup
- `02–03`: Ollama and Claude CLI agents
- `04`: Custom `BaseResponder` subclass example
- `05`: CUDA / GPU acceleration
- `06`: Pure transcription (`Spych.listen()`)
- `07–10`: Antigravity (agy), Codex, OpenCode, Claude SDK agents
- `11`: Multi-agent orchestration (Spanish/German translator via Ollama)
- `12`: Speaker TTS engine (Kokoro and Chatterbox)
- `13–14`: Bugfixes & parser unit tests

**Test pattern**:

```python
from spych import Spych, SpychWake

spych_object = Spych(whisper_model="base.en")

def on_wake(wake_word: str) -> None:
    text = spych_object.listen()
    print(f"Heard: {text}")

wake = SpychWake(
    wake_word_map={"hey computer": on_wake},
    terminate_words=["terminate"],
)
wake.start()
```

When adding a new feature, add a corresponding test file in `test/`. Run tests via `pytest` or `nox`.

---

## Coding Conventions

### Formatting

Always run `nox -s prettify` or `uv run python utils/prettify.py` before committing. This runs:
1. `autoflake` — removes unused imports
2. `black` — reformats code (configured in `pyproject.toml`)

### Type Hints

All functions and methods must have type hints.

- **Basic types**: use the built-in directly — `int`, `str`, `float`, `dict`, `list`, `tuple`
- **Class references**: use a string annotation — `"Spych"`, `"BaseResponder"`
- **Optional**: `Optional[str]` from `typing`
- **Union**: prefer `|` operator (`int | float`); use `Union[str, int]` from `typing` when needed
- **Complex dicts**: `dict[str, Any]` using `Any` from `typing`

```python
from typing import Any, Optional, Union

def listen(self, duration: Union[int, float, str] = 0, device_index: int = -1) -> str: ...
def respond(self, user_input: str) -> str: ...
wake_word_map: dict[str, callable]
spinner: Optional["CliSpinner"] = None
```

### Docstrings

Spych uses a custom structured docstring format. Every public class and method should include one.

```python
def method(self, param1: str, param2: int = 0) -> str:
    """
    Usage:

    - High-level description of what this method does.
    - Additional usage context if needed.

    Requires:

    - `param1`:
        - Type: str
        - What: Description of what this parameter does.

    Optional:

    - `param2`:
        - Type: int
        - What: Description of what this parameter does.
        - Default: 0
        - Note: Any constraint or caveat.

    Returns:

    - `return_value`:
        - Type: str
        - What: What the return value represents.

    Notes:

    - Any important implementation details or surprises.
    """
```

**Rules:**
- `Usage:` is always the first section — plain English, bullet points
- `Requires:` covers parameters without defaults; `Optional:` covers parameters with defaults
- Each parameter entry: backtick name, then `Type` → `What` → `Default` (optional params) → `Note` (if needed) sub-bullets
- `Returns:` describes the return value with the same sub-bullet format
- `Notes:` for implementation details, non-obvious behavior, or constraints
- Omit any section that doesn't apply (e.g., no `Requires:` if there are no required params)

### Other Rules

- **Runtime dependencies**: `claude_agent_sdk`, `faster-whisper`, `pvrecorder`, `numpy`, `requests`, `silero_vad`, `pygame`, `kokoro`, `huggingface_hub`, `chatterbox`, and the stdlib. Do not add new ones.
- **No unnecessary abstractions**: don't extract a shared helper unless the same logic appears 3+ times
- **DO NOT generate docs**: only the maintainer runs `./run.sh docs` at release time
