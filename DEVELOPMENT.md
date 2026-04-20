# spych: Development Guide

## Project Purpose

`spych` (pronounced "speech") is a lightweight, fully offline Python toolkit for voice-driven AI interactions. Core capabilities:

- **Wake word detection** — multi-threaded overlapping listener windows for continuous, low-latency detection
- **Voice transcription** — faster-whisper with optional Silero VAD gating for hands-free, auto-terminated recording
- **AI agent responders** — built-in support for Ollama, Claude Code (CLI and SDK), Gemini, Codex, and OpenCode
- **Multi-agent orchestration** — bind multiple responders to different wake words under a single listener session
- **Live transcription** — continuous VAD-gated transcription to `.txt` and/or `.srt` files
- **Extensible** — subclass `BaseResponder`, implement one method, and plug into the orchestration system

All core processing (VAD, transcription, wake word detection) runs fully offline on the local machine.

---

## Directory Layout (relevant files only)

```
spych/
  __init__.py              # Public exports: Spych, SpychWake, SpychLive, SpychOrchestrator, BaseResponder
  core.py                  # Spych — faster-whisper transcription engine with Silero VAD
  wake.py                  # SpychWake + SpychWakeListener — multi-threaded wake word detection
  responders.py            # BaseResponder — abstract base class for all AI agents
  orchestrator.py          # SpychOrchestrator — multi-agent coordinator with shared spinner
  live.py                  # SpychLive — continuous VAD-gated transcription to disk
  cli.py                   # CLI entry point (spych subcommands)
  cli_tools.py             # Theme, CliSpinner, CliPrinter — terminal UI utilities
  spinners.py              # Spinner frame definitions (BRAILLE, ARC, MOON, etc.)
  utils.py                 # Recorder, Notify (logging base), get_clean_audio_buffer
  agents/
    __init__.py            # Exports all agent factories and responder classes
    claude.py              # LocalClaudeCodeCLIResponder + LocalClaudeCodeSDKResponder + factories
    ollama.py              # OllamaResponder + ollama() factory
    gemini.py              # LocalGeminiCLIResponder + gemini_cli() factory
    codex.py               # LocalCodexCLIResponder + codex_cli() factory
    opencode.py            # LocalOpenCodeCLIResponder + opencode_cli() factory
    sdk_workers/
      claude_sdk_worker.py # Subprocess worker for Claude Agent SDK communication
test/
  NN_*.py                  # 11 numbered test scripts (01–11); run sequentially
utils/
  test.sh                  # Run all test/*.py files with python
  prettify.sh              # autoflake (unused imports) + black (line-length=88)
  docs.sh                  # Generate pdoc HTML docs — do NOT run unless releasing
docs/                      # Auto-generated pdoc HTML docs (do not edit manually)
pyproject.toml             # Package metadata and runtime dependencies
requirements.txt           # Dev dependencies (black, autoflake, pdoc, twine, build)
run.sh                     # Docker wrapper for all dev commands
Dockerfile                 # Multi-Python support (3.11–3.14); default is latest
```

---

## Development Commands

All commands use Docker via `./run.sh`:

| Command | What it does |
|---|---|
| `./run.sh test` | Run all tests inside Docker |
| `./run.sh test test/01_basic.py` | Run a specific test file in Docker |
| `./run.sh prettify` | Format with autoflake + black |
| `./run.sh docs` | Regenerate pdoc documentation |
| `./run.sh` | Drop into a Docker shell |

> **Note:** `./run.sh` requires a TTY. In non-interactive contexts (CI, background tasks) it will fail with "the input device is not a TTY". Ask the user to run it themselves.

**Test runner** (`utils/test.sh`): Runs every `*.py` file in `test/` with `python` sequentially. Each file is responsible for its own output.

**Docs**: **DO NOT generate docs**. Docs are regenerated and versioned at release time by the maintainer only.

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

**`BaseResponder`** (`responders.py`) — abstract agent base:
- Subclass and implement `respond(user_input: str) -> str` — the only required method
- Handles the full voice cycle: listen → transcribe → respond → print
- Optional hooks: `healthcheck()`, `on_before_respond()`, `on_after_respond()`
- `__call__() -> str`: runs one complete voice cycle

**`SpychOrchestrator`** (`orchestrator.py`) — multi-agent coordinator:
- Accepts a list of `OrchestratorEntry` dicts, each with `responder`, `wake_words`, and `terminate_words`
- Creates a single shared `CliSpinner` injected into all responders to prevent output conflicts
- Builds a flat `wake_word → responder.__call__` map and passes it to `SpychWake`
- `start()`: blocking; triggers the correct responder when its wake word is heard

**`SpychLive`** (`live.py`) — live transcription pipeline:
- Producer-consumer architecture with three threads: `VADRecorder` → `Transcriber` → `Writer`
- Writes to `.txt`, `.srt`, or both; maintains a context buffer (~128 words) for whisper `initial_prompt`
- Stoppable via keystroke, spoken terminate word, or `KeyboardInterrupt`

### Built-in Agents (`spych/agents/`)

| Class | Factory | Backend |
|---|---|---|
| `OllamaResponder` | `ollama()` | Local Ollama REST API |
| `LocalClaudeCodeCLIResponder` | `claude_code_cli()` | `claude` CLI subprocess |
| `LocalClaudeCodeSDKResponder` | `claude_code_sdk()` | Claude Agent SDK subprocess worker |
| `LocalGeminiCLIResponder` | `gemini_cli()` | `gemini` CLI subprocess |
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
│   ├── LocalGeminiCLIResponder
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
├── respond()                 # ← subclass implements this
├── on_after_respond()        # (hook — optional override)
├── on_response()             # Print response + elapsed time
└── wait_for_next_wake_word() # Print divider, restart spinner
```

---

## Test Structure

Tests are in `test/`. Each file is a standalone Python script: imports what it needs, runs a scenario, and prints its own output.

**Naming convention:** `NN_description.py` (zero-padded number ensures ordered execution)

**Rough groupings:**
- `01`: Wake word detection + listener setup
- `02–03`: Ollama and Claude CLI agents
- `04`: Custom `BaseResponder` subclass example
- `05`: CUDA / GPU acceleration
- `06`: Pure transcription (`Spych.listen()`)
- `07–10`: Gemini, Codex, OpenCode, Claude SDK agents
- `11`: Multi-agent orchestration (Spanish/German translator via Ollama)

**Test pattern** (most tests are interactive/manual; they demonstrate usage and run the agent loop):

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

When adding a new feature, add a corresponding `NN_*.py` test file. Tests are picked up automatically by `utils/test.sh`.

---

## Coding Conventions

### Formatting

Always run `./run.sh prettify` before committing. This runs:
1. `autoflake` — removes unused imports
2. `black` — reformats to line-length=88

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

- **No new runtime dependencies**: runtime code must only import from `claude_agent_sdk`, `faster-whisper`, `pvrecorder`, `numpy`, `requests`, `silero_vad`, and the stdlib
- **No unnecessary abstractions**: don't extract a shared helper unless the same logic appears 3+ times
- **DO NOT generate docs**: only the maintainer runs `./run.sh docs` at release time
