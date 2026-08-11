---
name: add-test
description: Create a new standalone Python test script in test/NN_description.py. Use when adding test coverage for a new feature, agent, wake word configuration, or live pipeline.
---

# Adding a Test Script

All tests in `spych` live in the `test/` directory. Each test is a standalone Python script designed to be run independently or via `pytest` / `nox`.

---

## 1. Naming & Execution Order

- **File Name Format**: `NN_description.py` (zero-padded 2-digit number) or `test_*.py`.
- **Execution Order**: `pytest` and `nox` discover and run test files automatically.
- **Next Available Number**: Check existing numbers in `test/` (e.g. `01` through `14`) and pick the next sequence integer (e.g. `15_my_test.py`).

```
test/
  01_basic.py
  02_ollama.py
  ...
  14_parser_unit.py
  15_new_feature.py   <-- New test
```

---

## 2. Standard Test Template

Tests demonstrate end-to-end usage or verify a specific module. A typical interactive test script sets up transcription and wake word listening or orchestrates an agent loop:

```python
"""
Test NN: Description of test scenario.
Run via: pytest test/NN_description.py or nox
"""

from spych import Spych, SpychWake


def test_main() -> None:
    print("Initializing Spych transcription engine...")
    spych_object = Spych(whisper_model="base.en")

    def on_wake(wake_word: str) -> None:
        print(f"Wake word detected: '{wake_word}'")
        text = spych_object.listen(duration="auto")
        print(f"Transcribed text: {text}")

    wake = SpychWake(
        wake_word_map={"hey computer": on_wake},
        terminate_words=["terminate", "stop"],
    )

    print("Starting wake word listener loop. Say 'hey computer' or 'terminate'...")
    wake.start()


if __name__ == "__main__":
    test_main()
```

---

## 3. Test Guidelines

- **Self-Contained**: The script should handle imports, configuration, output printing, and cleanup independently.
- **Printed Output**: Print clear, descriptive diagnostic logs to terminal stdout.
- **Non-blocking Options**: If testing a non-interactive mechanism, exit cleanly upon completion or catch `KeyboardInterrupt`.
- **Prettify**: Always run [prettify](../prettify/SKILL.md) after creating or editing test files.

---

## 4. Verification

Run the single test file:
```bash
pytest test/NN_description.py
```
Or run the full test suite across environments:
```bash
nox
```
