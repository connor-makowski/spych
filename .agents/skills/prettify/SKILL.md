---
name: prettify
description: Format spych codebase using autoflake and black via nox or python utils/prettify.py. Use when formatting, linting, cleaning up imports, or before committing.
---

# Code Formatting & Conventions

Before committing any Python changes in `spych`, format all files using the repo's formatting pipeline.

---

## 1. Running Prettify

### Via Nox:
```bash
nox -s prettify
```

### Direct / Local Shell:
```bash
python utils/prettify.py
```
or via `uv`:
```bash
uv run python utils/prettify.py
```

---

## 2. Formatting Rules & Tools

1. **`autoflake`**: Strips unused imports across `spych/`.
2. **`black`**: Formats all Python code in `spych/` and `test/` (configured in `pyproject.toml`).

---

## 3. Type Hinting Guidelines

All functions, methods, and class attributes in `spych` must include explicit type hints.

- Use Python 3.10+ union operator (`|`) for types: `int | float` (instead of `Union[int, float]`).
- Use `Optional[T]` from `typing` when appropriate: `Optional[str]`.
- For class self-references or forward references, use string annotations: `"Spych"`, `"BaseResponder"`.
- Use `dict[str, Any]` for complex dictionaries.

```python
from typing import Any, Optional


def listen(self, duration: int | float | str = 0, device_index: int = -1) -> str:
    ...


def respond(self, user_input: str) -> AgentResponse:
    ...


wake_word_map: dict[str, Any]
spinner: Optional["CliSpinner"] = None
```

---

## 4. Docstring Format

`spych` uses a custom structured docstring format for all public classes and methods:

```python
def method(self, param1: str, param2: int = 0) -> str:
    """
    Usage:

    - High-level description of what this method does.

    Requires:

    - `param1`:
        - Type: str
        - What: Description of parameter.

    Optional:

    - `param2`:
        - Type: int
        - What: Description of optional parameter.
        - Default: 0

    Returns:

    - `return_value`:
        - Type: str
        - What: What the return value represents.

    Notes:

    - Important implementation detail or caveat.
    """
```
