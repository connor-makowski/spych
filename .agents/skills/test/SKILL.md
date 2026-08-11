---
name: test
description: Run spych test suite using pytest or nox. Use when asked to run tests, verify changes, or test voice/agent components.
---

# Running Tests

Tests in `spych` are located in `test/` and are executed via `pytest` or `nox`.

---

## 1. Running Tests via `nox` (Multi-Environment)

Use `nox` to execute tests across Python environments:

```bash
# Run default pytest suite across supported Python versions (3.11–3.13)
nox

# Run a specific session for Python 3.12
nox -s tests-3.12
```

Note: `kokoro` is automatically installed on Python < 3.13 and `chatterbox-tts` is automatically installed on Python >= 3.13 via environment markers in `pyproject.toml`.

---

## 2. Running Tests Locally via `pytest` / `uv`

To run tests in your active virtual environment:

```bash
# Run all tests using pytest
pytest

# Run using uv
uv run pytest

# Run a specific test script
pytest test/12_speaker.py
```

---

## 3. Existing Test Registry

| File | Subsystem / Purpose | Notes |
|---|---|---|
| `01_basic.py` | Wake word detection & listener setup | Tests `SpychWake` construction and `wake()` dispatch |
| `02_ollama.py` | Ollama agent responder | Calls local Ollama API |
| `03_claude.py` | Claude CLI responder | Subprocess integration with `claude` |
| `05_cuda.py` | CUDA / GPU acceleration | Verifies Whisper GPU execution |
| `06_transcribe.py` | Standalone transcription | Pure `Spych.listen()` VAD testing |
| `07_agy.py` | Antigravity CLI responder | Subprocess integration with `agy` |
| `08_codex.py` | Codex CLI responder | Subprocess integration with `codex` |
| `09_opencode.py` | OpenCode CLI responder | Subprocess integration with `opencode` |
| `10_claude_sdk.py` | Claude Agent SDK responder | SDK subprocess worker test |
| `11_translator.py` | Multi-agent translation | Spanish/German translation via Ollama |
| `12_speaker.py` | Speaker TTS engine | Tests Kokoro and Chatterbox backends |
| `13_bugfixes.py` | Regression fixes | Bugfix regression assertions |
| `14_parser_unit.py` | LLM JSON output parser | Unit test coverage for `parse_output` |
| `15_cli_dispatch.py` | CLI dispatch (`spych/cli.py`) | Runs `main()` for every subcommand with blocking loops stubbed out; catches missing-import / undefined-`args.*` bugs |

---

## 4. Test Failure Checklist

If a test fails:
1. Verify audio device configuration (`pvrecorder` hardware index or mic input).
2. Check LLM backend availability (e.g. `ollama serve` running or CLI binary available in PATH).
3. Inspect tracebacks and logs carefully before making code edits.
4. Rerun formatting ([prettify](../prettify/SKILL.md)) after applying fixes.
