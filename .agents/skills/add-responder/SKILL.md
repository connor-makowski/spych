---
name: add-responder
description: Add a new AI agent responder (Ollama, Claude, Antigravity, Codex, OpenCode, or custom LLM) by subclassing BaseResponder. Use when implementing a new agent responder, creating a custom LLM responder, or adding built-in agent support.
---

# Adding an Agent Responder

All AI agents in `spych` subclass `BaseResponder` (defined in `spych/responders.py`). Responders handle prompt formatting, LLM execution, JSON response parsing, and standard voice lifecycle hooks.

---

## 1. Class Implementation Pattern

Create your responder in `spych/agents/<agent_name>.py`:

```python
from spych.responders import AgentResponse, BaseResponder


class LocalMyAgentCLIResponder(BaseResponder):
    """
    Usage:

    - Responder for interacting with MyAgent CLI.

    Requires:

    - `spych_object`:
        - Type: Spych
        - What: Transcriber instance.

    Optional:

    - `model`:
        - Type: str
        - What: MyAgent model name.
        - Default: "default-model"
    """

    def __init__(
        self,
        spych_object: "Spych",
        model: str = "default-model",
        use_speaker: bool = False,
        speaker_voice: str = "af_heart",
        response_style: str = "concise",
    ) -> None:
        super().__init__(
            spych_object=spych_object,
            use_speaker=use_speaker,
            speaker_voice=speaker_voice,
            response_style=response_style,
        )
        self.model = model

    def respond(self, user_input: str) -> AgentResponse:
        """
        Usage:

        - Send formatted prompt to LLM and return structured AgentResponse.

        Requires:

        - `user_input`:
            - Type: str
            - What: Transcribed text from speech-to-text engine.

        Returns:

        - `response`:
            - Type: AgentResponse
            - What: Structured response containing full text, spoken summary, and feedback boolean.
        """
        formatted_prompt = self.format_prompt(user_input)

        # 1. Call your LLM CLI, REST API, or SDK worker here
        raw_output = self._call_llm(formatted_prompt)

        # 2. Parse LLM response into AgentResponse (handles JSON codeblocks & raw JSON)
        return self.parse_output(raw_output)

    def _call_llm(self, prompt: str) -> str:
        # Implementation details for invoking the LLM model/subprocess/API
        ...
```

---

## 2. Structured Response Requirement

The LLM output is parsed by `self.parse_output(raw_output)`. Your prompt formatted via `self.format_prompt(user_input)` requests a JSON payload with:
- `response`: Full markdown text printed to stdout.
- `summary`: Short, spoken-friendly summary read aloud by `Speaker`.
- `requires_user_feedback`: `True` if the response ends with a question, triggering `spoken_follow_up_loop`.

---

## 3. Factory Function Pattern

Define a factory function in the same module (`spych/agents/<agent_name>.py`) that creates a `Spych` instance, wraps it in the responder, and launches a `SpychOrchestrator`:

```python
from spych.core import Spych
from spych.orchestrator import OrchestratorEntry, SpychOrchestrator


def myagent(
    wake_words: list[str] | None = None,
    model: str = "default-model",
    spych_kwargs: dict | None = None,
    spych_wake_kwargs: dict | None = None,
    **responder_kwargs,
) -> SpychOrchestrator:
    if wake_words is None:
        wake_words = ["hey agent", "computer"]

    spych_object = Spych(**(spych_kwargs or {}))
    responder = LocalMyAgentCLIResponder(
        spych_object=spych_object,
        model=model,
        **responder_kwargs,
    )

    entry = OrchestratorEntry(responder=responder, wake_words=wake_words)
    return SpychOrchestrator([entry], spych_wake_kwargs=spych_wake_kwargs)
```

---

## 4. Export & Registration

1. **Export in `spych/agents/__init__.py`**:
   ```python
   from spych.agents.myagent import LocalMyAgentCLIResponder, myagent
   ```
2. **Export in `spych/__init__.py`**:
   Add `LocalMyAgentCLIResponder` and `myagent` to `__all__`.
3. **Register CLI flag** (if applicable) in `spych/cli.py`.

---

## 5. Testing & Verification

1. Add a numbered test script in `test/` (e.g., `test/13_myagent.py`). See [add-test](../add-test/SKILL.md).
2. Run test: `pytest test/13_myagent.py` or `nox -s tests`.
3. Run formatting: [prettify](../prettify/SKILL.md).
