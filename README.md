# Spych
[![PyPI version](https://badge.fury.io/py/spych.svg)](https://badge.fury.io/py/spych)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI Downloads](https://img.shields.io/pypi/dm/spych.svg?label=PyPI%20downloads)](https://pypi.org/project/spych/)

**Spych** (pronounced "speech"): talk to your computer like its your personal assistant without sending your voice to the cloud.

A lightweight, fully offline Python toolkit for wake word detection, audio transcription, and AI integrations. Built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and [PvRecorder](https://github.com/Picovoice/pvrecorder).

- **Fully offline**: no API keys, no cloud calls, no eavesdropping
- **Multi-threaded wake word detection**: overlapping listener windows so you rarely miss a trigger
- **Multiple wake words**: map different words to different actions in one listener
- **Built-in agents**: for [Ollama](https://ollama.com) and [Claude Code](https://docs.anthropic.com/en/docs/claude-code)

---

# Setup

Requires Python 3.11+. Download it [here](https://www.python.org/downloads/) if needed.

## Installation
```
pip install spych
```

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

Voice-control Claude Code directly from your voice.

```python
from spych.agents import claude_code_cli

# Say "hey claude" to trigger
claude_code_cli()
```

> 💡 **Pro tip:** Saying "Hey Llama" or "Hey Claude" tends to trigger more reliably than just the bare wake word.

Both agents accept a `terminate_words` list (default: `["terminate"]`). Say the word or use `ctrl+c` to stop the listener cleanly.

### Agent Parameters

| Parameter | `ollama` default | `claude_code_cli` default | Description |
|---|---|---|---|
| `wake_words` | `["llama", "ollama", "lama"]` | `["claude", "clod", "cloud", "clawed"]` | Words that trigger the agent |
| `terminate_words` | `["terminate"]` | `["terminate"]` | Words that stop the listener |
| `model` | `"llama3.2:latest"` | - | Ollama model name |
| `listen_duration` | `5` | `5` | Seconds to listen after wake word |
| `continue_conversation` | - | `True` | Resume the most recent Claude session |
| `history_length` | `10` | - | Past interactions to include in Ollama context |
| `host` | `"http://localhost:11434"` | - | Ollama instance URL |
| `spych_kwargs` | - | - | Extra kwargs passed to `Spych` |
| `spych_wake_kwargs` | - | - | Extra kwargs passed to `SpychWake` |

---

# Building Your Own Agent

Not using Ollama or Claude? No problem. Subclass `BaseResponder`, implement `respond`, and you're done. Spych handles the rest: listening, transcription, spinner UI, timing, error handling, all of it.
```python
from spych.responders import BaseResponder

class MyResponder(BaseResponder):
    def respond(self, user_input: str) -> str:
        return f"'{self.name}' heard: {user_input}"
```

A complete working example with a custom wake word:
```python
from spych.responders import BaseResponder
from spych import Spych, SpychWake

class MyResponder(BaseResponder):
    def respond(self, user_input: str) -> str:
        return f"'{self.name}' heard: {user_input}"

my_responder = MyResponder(
    spych_object = Spych(whisper_model="base.en"),
    listen_duration=5,
    name="TestResponder"
)

wake_object = SpychWake(
    wake_word_map={"test": my_responder},
    whisper_model="tiny.en",
    terminate_words=["terminate"]
)
my_responder.ready_message(wake_words = ["test"], terminate_words = ["terminate"])
wake_object.start()
```

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

<!-- ## Multiple Wake Words

Map different words to different callbacks all in one listener.
```python
from spych import SpychWake, Spych
from spych.agents import OllamaResponder, LocalClaudeCodeCLIResponder

spych = Spych(whisper_model="base.en", whisper_device="cpu")

wake = SpychWake(
    wake_word_map={
        "llama": OllamaResponder(spych, model="llama3.2:latest"),
        "claude": LocalClaudeCodeCLIResponder(spych),
    },
    whisper_model="tiny.en",
    terminate_words=["terminate"]
)

wake.start()
```

See: https://connor-makowski.github.io/spych/spych/wake.html -->

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