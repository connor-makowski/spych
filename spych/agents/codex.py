from spych.core import Spych
from spych.wake import SpychWake
from spych.responders import BaseResponder
from spych.cli import CliColor, CliPrinter
from typing import Optional, Any
import subprocess
import json
import time


class LocalCodexCLIResponder(BaseResponder):
    def __init__(
        self,
        spych_object: "Spych",
        continue_conversation: bool = True,
        listen_duration: int | float = 5,
        name: Optional[str] = "Codex",
        show_tool_events: bool = True,
    ) -> None:
        """
        Usage:

        - A responder that pipes transcribed audio into the Codex CLI (`codex`)
          via a subprocess and returns the final response string.
          Fires live tool-call events as the subprocess streams them.

        Requires:

        - `spych_object`:
            - Type: Spych
            - What: An initialized Spych instance used to record and transcribe audio

        Optional:

        - `continue_conversation`:
            - Type: bool
            - What: Whether to pass `--continue` to reuse the most recent session
            - Default: True

        - `listen_duration`:
            - Type: int | float
            - What: The number of seconds to listen for after the wake word is detected
            - Default: 5

        - `name`:
            - Type: str
            - What: A custom name for the responder to use in printed messages
            - Default: "Codex"

        - `show_tool_events`:
            - Type: bool
            - What: Whether to print tool start/end events in the CLI as they arrive from the subprocess
            - Default: True

        Notes:

        - Uses `--json` to stream intermediate tool call events
        - Codex CLI must be installed and authenticated before use
          (run `codex` once interactively to complete OAuth / API-key setup)
        """
        super().__init__(
            spych_object=spych_object,
            listen_duration=listen_duration,
            name=name,
        )
        self.continue_conversation = continue_conversation
        self.show_tool_events = show_tool_events
        self.first_call = True
        self._last_session_id: Optional[str] = None

    def respond(self, user_input: str) -> str:
        """
        Usage:

        - Pipes the transcribed user input into `codex` with `--json`
          and returns the final response after all tool calls have completed.
          Fires tool events live as they arrive.

        Requires:

        - `user_input`:
            - Type: str
            - What: The transcribed text from the user's audio input

        Returns:

        - `response`:
            - Type: str
            - What: The final response string from Codex CLI
        """
        is_first = self.first_call
        self.first_call = False

        # `codex --json -q <prompt>` runs non-interactively and streams JSON events.
        # `-q` / `--quiet` suppresses the spinner so stdout is clean JSON lines.
        cmd = ["codex", "exec", "--json", user_input]

        if self.continue_conversation:
            if self._last_session_id:
                cmd.extend(["--session", self._last_session_id])
            elif not is_first:
                cmd.append("--continue")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        final_result = ""
        # item_id -> (tool_name, start_time)  for function-call items
        active_tools: dict[str, tuple[str, float]] = {}

        for raw_line in proc.stdout:
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            etype = event.get("type")

            # --- Real Codex stream event types ---
            # "thread.started"   – new thread; carries thread_id
            # "turn.started"     – model turn beginning
            # "item.completed"   – one discrete item finished; item.type can be:
            #                        "agent_message"  – assistant text (item.text)
            #                        "function_call"  – tool invocation (item.name / item.arguments)
            #                        "function_output"– tool result    (item.name / item.output)
            #                        "error"          – error message  (item.message)
            # "turn.completed"   – turn done; carries usage stats

            if etype == "thread.started":
                self._last_session_id = event.get("thread_id")

            elif etype == "item.completed":
                item = event.get("item", {})
                itype = item.get("type")
                item_id = item.get("id", "")

                if itype == "agent_message":
                    text = item.get("text", "")
                    if text:
                        final_result = text

                elif itype == "function_call":
                    tool_name = item.get("name", "unknown_tool")
                    tool_input = item.get("arguments", "{}")
                    active_tools[item_id] = (tool_name, time.time())
                    if self.show_tool_events:
                        self.tool_event(tool_name, tool_input, is_running=True)

                elif itype == "function_output":
                    tool_name = item.get("name", "")
                    # Match by name when id isn't available
                    matched_id = item_id
                    if matched_id not in active_tools:
                        matched_id = next(
                            (k for k, (n, _) in active_tools.items() if n == tool_name),
                            None,
                        )
                    if matched_id and matched_id in active_tools:
                        _, start = active_tools.pop(matched_id)
                        elapsed = time.time() - start
                        if self.show_tool_events:
                            self.tool_event(tool_name, "done", is_running=False, elapsed=elapsed)

                elif itype == "error":
                    final_result = f"Error: {item.get('message', 'unknown error')}"

            # turn.completed carries usage but no new text — nothing to handle

        proc.wait()
        return final_result


def codex_cli(
    wake_words: list[str] = ["codex"],
    terminate_words: list[str] = ["terminate"],
    listen_duration: int | float = 5,
    continue_conversation: bool = True,
    show_tool_events: bool = True,
    spych_kwargs: Optional[dict[str, Any]] = None,
    spych_wake_kwargs: Optional[dict[str, Any]] = None,
) -> None:
    """
    Usage:

    - Starts a wake word listener that pipes detected speech into the Codex CLI

    Optional:

    - `wake_words`:
        - Type: list[str]
        - What: A list of wake words that each trigger the Codex CLI responder
        - Default: ["codex"]
        - Note: All wake words in this list map to the same LocalCodexCLIResponder
          instance, sharing conversation history across triggers

    - `terminate_words`:
        - Type: list[str]
        - What: A list of terminate words that each trigger the termination of the Codex CLI responder
        - Default: ["terminate"]

    - `listen_duration`:
        - Type: int | float
        - What: The number of seconds to listen for after the wake word is detected
        - Default: 5

    - `continue_conversation`:
        - Type: bool
        - What: Whether to pass `--continue` / `--session` to reuse the most recent session
        - Default: True

    - `show_tool_events`:
        - Type: bool
        - What: Whether to print tool start/end events in the CLI as they arrive from the subprocess
        - Default: True

    - `spych_kwargs`:
        - Type: dict
        - What: Additional keyword arguments to pass to the Spych constructor
        - Default: None

    - `spych_wake_kwargs`:
        - Type: dict
        - What: Additional keyword arguments to pass to the SpychWake constructor
        - Default: None
    """
    # Spych Object
    spych_kwargs = {"whisper_model": "base.en", **(spych_kwargs or {})}
    spych_object = Spych(**spych_kwargs)

    # Responder Object
    responder = LocalCodexCLIResponder(
        spych_object=spych_object,
        continue_conversation=continue_conversation,
        listen_duration=listen_duration,
        show_tool_events=show_tool_events,
    )

    # SpychWake Object
    spych_wake_kwargs = {
        "whisper_model": "base.en",
        "on_terminate": responder.on_terminate,
        "wake_word_map": {word: responder for word in wake_words},
        "terminate_words": terminate_words,
        **(spych_wake_kwargs or {}),
    }
    spych_wake_object = SpychWake(**spych_wake_kwargs)

    # Fire ready message and start wake listener
    responder.ready_message(wake_words=wake_words, terminate_words=terminate_words)
    spych_wake_object.start()