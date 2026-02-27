"""
# Spych
[![PyPI version](https://badge.fury.io/py/spych.svg)](https://badge.fury.io/py/spych)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI Downloads](https://img.shields.io/pypi/dm/spych.svg?label=PyPI%20downloads)](https://pypi.org/project/spych/)

Spych (pronounced "speech") — a lightweight, fully offline Python speech-to-text toolkit for wake word detection and audio transcription. Built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and [PvRecorder](https://github.com/Picovoice/pvrecorder).

- Fully offline -> no API keys, no cloud calls, no internet required
- Multi-threaded wake word detection that never misses a trigger between recording windows
- Multiple wake words, each mapped to a different action
- Built-in agents for [Ollama](https://ollama.com) and [Claude Code](https://docs.anthropic.com/en/docs/claude-code)

# Setup

Make sure you have Python 3.11.x (or higher) installed on your system. You can download it [here](https://www.python.org/downloads/).

## Installation

```
pip install spych
```

## Quick Start: Voice Agents

The fastest way to get started is via `spych.agents`. These one-liners handle everything — wake word detection, transcription, and passing your speech to the target agent.

### Ollama

Requires [Ollama](https://ollama.com) to be installed and running locally. Pull a model first with `ollama pull llama3.2:latest`.

```python
from spych.agents import ollama

# Say "llama" or "ollama" to trigger
ollama(model="llama3.2:latest", whisper_device="cuda")
```

### Claude Code CLI

Requires [Claude Code](https://docs.anthropic.com/en/docs/claude-code) to be installed in your **terminal** and authenticated. Verify with `claude --version` in your terminal. Fun side hint, you can run claude code with ollama if you want a fully offline experience.

```python
from spych.agents import claude_code_cli

# Say "claude" to trigger
claude_code_cli(whisper_device="cuda")
```

Both agents support a `terminate_words` list (default: `["terminate"]`) — saying a terminate word will stop the listener cleanly.

| Parameter | `ollama` default | `claude_code_cli` default | Description |
|---|---|---|---|
| `whisper_device` | `"cpu"` | `"cpu"` | `"cpu"` or `"cuda"` |
| `wake_words` | `["llama", "ollama"]` | `["claude"]` | Words that trigger the agent |
| `terminate_words` | `["terminate"]` | `["terminate"]` | Words that stop the listener |
| `model` | `"llama3.2:latest"` | — | Ollama model name |
| `listen_duration` | `5` | `5` | Seconds to listen for after wake word |
| `continue_conversation` | — | `True` | Whether to reuse the most recent session in Claude CLI |
| `history_length` | `10` | — | Number of past interactions to include in the prompt sent to Ollama |
| `host` | `"http://localhost:11434"` | — | URL of the Ollama instance to connect to |

### Want a different agent?
No problem! You can build your own custom agent by subclassing `BaseResponder` and passing an instance to `SpychWake`. See the API reference below for details.

Think others would find your agent useful? Open a PR or make a feature request via a git issue.
---

# Wake Word Detection + Transcription

If you want more control, use `SpychWake` and `Spych` directly.

## Listen and Transcribe

`Spych` records from the microphone and returns a transcription string.

```python
from spych import Spych

spych_object = Spych(
    whisper_model="base.en",
    whisper_device="cuda",  # or "cpu"
    whisper_compute_type="int8",
)

transcription = spych_object.listen(duration=5)
print(transcription)
```
See: https://connor-makowski.github.io/spych/spych/core.html


## Wake Word Detection

`SpychWake` listens continuously for one or more wake words and fires a callback when detected.

```python
from spych import SpychWake, Spych

spych_object = Spych(whisper_model="base.en", whisper_device="cuda", whisper_compute_type="int8")

def on_wake():
    print("Wake word heard! Listening for 5 seconds...")
    print(spych_object.listen(duration=5))

wake_object = SpychWake(
    wake_word_map={"speech": on_wake},
    whisper_model="tiny.en",
    whisper_device="cuda",
    whisper_compute_type="int8",
)

wake_object.start()
```
See: https://connor-makowski.github.io/spych/spych/wake.html

## Multiple Wake Words

Map different wake words to different callbacks in a single listener.

```python
from spych import SpychWake, Spych
from spych.responders import OllamaResponder, LocalClaudeCodeCLIResponder

spych_object = Spych(whisper_model="base.en", whisper_device="cuda", whisper_compute_type="int8")
wake_object = SpychWake(
    wake_word_map={
        "llama": OllamaResponder(spych_object, model="llama3.2:latest"),
        "claude": LocalClaudeCodeCLIResponder(spych_object),
    },
    whisper_model="tiny.en",
    whisper_device="cuda",
    whisper_compute_type="int8",
    terminate_words=["terminate"]
)
wake_object.start()
```
See: https://connor-makowski.github.io/spych/spych/wake.html

---

# API Reference

## For a full API reference, including all parameters and methods,
See: https://connor-makowski.github.io/spych/spych.html

## Responders

Responders need additional explanations. Responders handle the listen-transcribe-respond cycle and are callable, making them suitable as `wake_word_map` values. All responders inherit from `BaseResponder`.

### `BaseResponder`

Subclass this to build a custom integration. Only `respond` needs to be implemented.

```python
from spych.responders import BaseResponder

class MyResponder(BaseResponder):
    def respond(self, user_input: str) -> str:
        # Your custom logic here
        # For exmample, you may pass this to an API, run some code, or trigger a local action
        return "response"
```

A full implementation that uses `test` as a wake word would look like:

```python
from spych.responders import BaseResponder
from spych import Spych, SpychWake

class MyResponder(BaseResponder):
    # A custom init function to set up any necessary variables or connections for your responder
    def __init__(self, spych_object):
        super().__init__(spych_object)
        # You can also customize the responder's name
        self.name = "Custom Responder"

    def respond(self, user_input: str) -> str:
        # Your custom logic here
        return "I am a custom responder and I heard: " + user_input

spych_object = Spych(whisper_model="base.en", whisper_device="cuda", whisper_compute_type="int8")
wake_object = SpychWake(
    wake_word_map={"test": MyResponder(spych_object)},
    whisper_model="tiny.en",
    whisper_device="cuda",
    whisper_compute_type="int8",
    terminate_words=["terminate"]
)
print("Starting wake listener. Say 'test' to trigger the MyResponder function.")
wake_object.start()
```

# Support

## Bug Reports and Feature Requests

If you find a bug or are looking for a new feature, please open an issue on GitHub.

## Need Help?

If you need help, please open an issue on GitHub.

# Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Making Changes

1) Fork the repo and clone it locally.
2) Make your modifications.
3) Run tests and make sure they pass.
4) Only commit relevant changes and add clear commit messages.
    - Atomic commits are preferred.
5) Submit a pull request.

## Virtual Environment

- Create a virtual environment: `python3.11 -m venv venv`
- Activate: `source venv/bin/activate`
- Install dev requirements: `pip install -r requirements/dev.txt`
- Run tests: `./utils/test.sh`"""

from .core import Spych
from .wake import SpychWake
