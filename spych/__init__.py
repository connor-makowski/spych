"""
# Spych
[![PyPI version](https://badge.fury.io/py/spych.svg)](https://badge.fury.io/py/spych)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI Downloads](https://img.shields.io/pypi/dm/spych.svg?label=PyPI%20downloads)](https://pypi.org/project/spych/)

Spych (pronounced speech) A lightweight, fully offline Python speech to text toolkit for wake word detection and audio transcription. Built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and [PvRecorder](https://github.com/Picovoice/pvrecorder).

# Setup

Make sure you have Python 3.11.x (or higher) installed on your system. You can download it [here](https://www.python.org/downloads/).

## Installation

```
pip install spych
```

### Dependencies

- `faster-whisper` — offline transcription via OpenAI Whisper models (CUDA or CPU)
- `pvrecorder` — cross-platform microphone capture

## Basic Usage

```python
from spych import SpychWake, Spych

wake_object = SpychWake(
    wake_word="speech",
    whisper_model="tiny.en",
    whisper_device="cuda", # You can comment this out if you do not have a compatible GPU
    whisper_compute_type="int8", 
)

spych_object = Spych(
    whisper_model="base.en",
    whisper_device="cuda", # You can comment this out if you do not have a compatible GPU
    whisper_compute_type="int8",
)

def on_wake():
    print("Wake word heard! Listening for 5 seconds...")
    output = spych_object.listen(duration=5)
    print(f"Transcription: {output}")

print("Starting wake listener. Say 'spych' (pronounced speech) to trigger the on_wake function.")
wake_object.start(on_wake_fn=on_wake)
```

## Getting Started

`spych` exposes two primary classes: `Spych` for transcription and `SpychWake` for wake word detection.

`Spych` wraps a faster-whisper model and provides a `listen` method that records from the microphone and returns a transcription string.

`SpychWake` runs multiple overlapping listener threads in the background, each recording a short audio clip and checking for the wake word. When detected, it locks the listeners and fires the provided `on_wake_fn` callback. This overlapping approach ensures continuous coverage without gaps.

Both classes inherit from `Notify`, which provides a consistent internal logging and exception system.

## Why use Spych?

- `spych` is fully offline. No API keys, no cloud calls, no internet required
- `spych` uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for fast, accurate transcription on both CPU and GPU
- `spych` uses an overlapping multi-threaded listener design to ensure the wake word is never missed between recording windows
- `spych` is designed to be minimal and easy to integrate into any voice-driven application

# Classes

## `Spych`

Handles microphone recording and transcription.

```python
from spych import Spych

spych_object = Spych(
    whisper_model="base.en",   # any faster-whisper compatible model
    whisper_device="cpu",      # "cpu" or "cuda"
    whisper_compute_type="int8",
)
```

### `listen`

Record from the microphone and return a transcription string.

```python
transcription = spych_object.listen(duration=5, device_index=-1)
print(transcription)
```

- `duration` (default: `5`): How many seconds to record
- `device_index` (default: `-1`): Microphone device index. `-1` uses the system default.

## `SpychWake`

Listens continuously for a wake word using overlapping threads, then fires a callback.

```python
from spych import SpychWake

wake_object = SpychWake(
    wake_word="speech",
    wake_listener_count=3,
    wake_listener_time=2,
    wake_listener_max_processing_time=0.5,
    device_index=-1,
    whisper_model="tiny.en",
    whisper_device="cpu",
    whisper_compute_type="int8",
)
```

| Parameter | Default | Description |
|---|---|---|
| `wake_word` | required | The word to listen for (case-insensitive) |
| `wake_listener_count` | `3` | Number of concurrent listener threads |
| `wake_listener_time` | `2` | Recording duration per listener in seconds |
| `wake_listener_max_processing_time` | `0.5` | Max expected transcription time used to stagger thread launches |
| `device_index` | `-1` | Microphone device index as defined by PvRecorder(`-1` = system default) |
| `whisper_model` | `"tiny"` | faster-whisper model name |
| `whisper_device` | `"cpu"` | `"cpu"` or `"cuda"` |
| `whisper_compute_type` | `"int8"` | Compute type for the whisper model |

### `start`

Begin listening. Blocks until `KeyboardInterrupt` or `stop()` is called.

```python
wake_object.start(on_wake_fn=my_callback)
```

- `on_wake_fn`: A no-argument callable that is executed each time the wake word is detected.

### `stop`

Stop all listeners and exit the `start` loop.

```python
wake_object.stop()
```

# Examples

## Basic Wake + Transcribe

```python
from spych import SpychWake, Spych

wake_object = SpychWake(
  wake_word="speech", 
  whisper_model="tiny.en",  # any faster-whisper compatible model
  whisper_device="cuda",    # "cpu" or "cuda"
  whisper_compute_type="int8"
)
spych_object = Spych(
  whisper_model="base.en",  # any faster-whisper compatible model
  whisper_device="cuda",    # "cpu" or "cuda"
  whisper_compute_type="int8"
)

def on_wake():
    print("Wake word heard! Listening for 5 seconds...")
    output = spych_object.listen(duration=5)
    print(f"Transcription: {output}")

print("Starting wake listener. Say 'spych' (pronounced speech) to trigger the on_wake function.")
wake_object.start(on_wake_fn=on_wake)
```

## Wake + Ollama Agent

Combine spych with a locally running [Ollama](https://ollama.com) instance for a fully offline voice assistant.

```python
from spych import SpychWake
from spych import Spych
import requests

wake_object = SpychWake(
    wake_word="speech",
    whisper_model="tiny.en",  # any faster-whisper compatible model
    whisper_device="cuda",    # "cpu" or "cuda"
    whisper_compute_type="int8",
)

spych_object = Spych(
    whisper_model="base.en",  # any faster-whisper compatible model
    whisper_device="cuda",    # "cpu" or "cuda"
    whisper_compute_type="int8"
)

class AgentResponder:
    def __init__(self, model="llama3.2:latest", history_length=10):
        self.history = []
        self.model = model
        self.history_length = history_length

    def ask_ollama(self, prompt):
        self.history.append({"role": "user", "content": prompt})
        prompt = "\n".join([f"{entry['role'].capitalize()}: {entry['content']}" for entry in self.history]) + "\nAssistant:"
        output = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
        )
        response = output.json().get("response", "").strip()
        self.history.append({"role": "assistant", "content": response})
        self.history = self.history[-self.history_length*2:]  # Keep only the last N interactions
        return response

    def __call__(self):
        print("Listening to you for 5 seconds...")
        user_input = spych_object.listen(duration=5)
        print(f"You: {user_input}")
        agent_output = self.ask_ollama(user_input)
        print(f"Agent: {agent_output}")
        print("=" * 20)


print("=" * 20)
print("Starting wake listener. Say 'spych' (pronounced speech) to trigger the on_wake function.")
print("=" * 20)
wake_object.start(on_wake_fn=AgentResponder())
```

# Support

## Bug Reports and Feature Requests

If you find a bug or are looking for a new feature, please open an issue on GitHub.

## Need Help?

If you need help, please open an issue on GitHub.

# Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Development

For development, we recommend using a virtual environment.

## Making Changes

1) Fork the repo and clone it locally.
2) Make your modifications.
3) Run tests and make sure they pass.
4) Only commit relevant changes and add clear commit messages.
    - Atomic commits are preferred.
5) Submit a pull request.

## Virtual Environment

- Create a virtual environment
    - `python3.11 -m venv venv`
- Activate the virtual environment
    - `source venv/bin/activate`
- Install the development requirements
    - `pip install -r requirements/dev.txt`
- Run Tests
    - `./utils/test.sh`"""

from .core import Spych
from .wake import SpychWake
