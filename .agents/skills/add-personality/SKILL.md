---
name: add-personality
description: Add or modify voice agent personality presets in PERSONALITIES. Use when defining new character presets, wake word triggers, speaker voices, or response styles.
---

# Adding Personality Presets

`spych` includes preset agent personalities that configure wake words, TTS voices, speaker usage, and response styles under unified preset names.

---

## 1. Personality Schema (`spych/utils.py`)

Personalities are defined in the `PERSONALITIES` dictionary in `spych/utils.py`:

```python
PERSONALITIES: dict[str, dict[str, Any]] = {
    "jarvis": {
        "name": "Jarvis",
        "wake_words": ["hey jarvis", "jarvis"],
        "speaker_voice": "af_heart",
        "use_speaker": True,
        "response_style": "concise",
    },
    "custom_preset": {
        "name": "CustomPreset",
        "wake_words": ["hey custom", "computer"],
        "speaker_voice": "af_bella",
        "use_speaker": True,
        "response_style": "technical",
    },
}
```

---

## 2. Response Styles (`spych/utils.py`)

To add or customize the prompt style associated with a personality:

1. Locate `get_response_style(style_name: str)` in `spych/utils.py`.
2. Add or update the prompt template in the `styles` map:

```python
styles = {
    "concise": "Respond concisely in 1-2 direct sentences.",
    "technical": "Provide detailed technical breakdown with clear markdown structure.",
    "my_custom_style": "Speak in a formal, structured tone suited for voice playback.",
}
```

---

## 3. CLI Integration (`spych/cli.py`)

The CLI accepts `--personality <name>` (e.g. `spych chat --personality jarvis`).

When `--personality` is supplied, `spych/cli.py` looks up the preset in `PERSONALITIES` and applies its settings as default arguments prior to parsing explicit overrides.

---

## 4. Verification

1. Create a test script or test via CLI:
   ```bash
   spych chat --personality custom_preset
   ```
2. Verify wake words, TTS voice selection, and LLM output style.
3. Run [prettify](../prettify/SKILL.md).
