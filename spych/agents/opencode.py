from spych.core import Spych
from spych.wake import SpychWake
from spych.responders import BaseResponder
from typing import Optional, Any
import subprocess
import json
import time


class LocalOpenCodeCLIResponder(BaseResponder):
    def __init__(
        self,
        spych_object: "Spych",
        continue_conversation: bool = True,
        listen_duration: int | float = 5,
        name: Optional[str] = "OpenCode",
        show_tool_events: bool = True,
        model: Optional[str] = None,
    ) -> None:
        """
        Usage:

        - A responder that pipes transcribed audio into the OpenCode CLI
          (`opencode run --format json`) via a subprocess and returns the
          final response string. Fires live tool-call events as the subprocess
          streams them.

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
            - Default: "OpenCode"

        - `show_tool_events`:
            - Type: bool
            - What: Whether to print tool start/end events in the CLI as they arrive
            - Default: True

        - `model`:
            - Type: str
            - What: Model to use in provider/model format (e.g. "anthropic/claude-sonnet-4-5")
            - Default: None (uses opencode default)

        Notes:

        - Uses `--format json` to stream newline-delimited JSON events
        - OpenCode CLI must be installed and authenticated before use
        """
        super().__init__(
            spych_object=spych_object,
            listen_duration=listen_duration,
            name=name,
        )
        self.continue_conversation = continue_conversation
        self.show_tool_events = show_tool_events
        self.model = model
        self.first_call = True
        self._last_session_id: Optional[str] = None

    def respond(self, user_input: str) -> str:
        """
        Usage:

        - Pipes the transcribed user input into `opencode run --format json`
          and returns the final response after all tool calls have completed.
          Fires tool events live as they arrive.

        Requires:

        - `user_input`:
            - Type: str
            - What: The transcribed text from the user's audio input

        Returns:

        - `response`:
            - Type: str
            - What: The final response string from OpenCode CLI
        """
        is_first = self.first_call
        self.first_call = False

        cmd = ["opencode", "run", "--format", "json", user_input]

        if self.continue_conversation:
            if self._last_session_id:
                cmd.extend(["--session", self._last_session_id])
            elif not is_first:
                cmd.append("--continue")

        if self.model:
            cmd.extend(["--model", self.model])

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        final_result = ""
        # callID -> (tool_name, start_time)
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
            part = event.get("part", {})

            # Capture session ID from any event for conversation continuity
            session_id = event.get("sessionID")
            if session_id:
                self._last_session_id = session_id

            # --- OpenCode stream event types ---
            # "step_start"  – model turn beginning
            # "tool_use"    – tool invocation; part.state.status == "completed" means done
            #                   part.callID, part.tool, part.state.input / part.state.output
            # "text"        – assistant text delta (part.text); last one before stop is final answer
            # "step_finish" – turn done; part.reason == "stop" signals final turn

            if etype == "tool_use":
                call_id = part.get("callID", "")
                tool_name = part.get("tool", "unknown_tool")
                state = part.get("state", {})
                status = state.get("status")

                if status == "completed":
                    if call_id in active_tools:
                        _, start = active_tools.pop(call_id)
                        elapsed = time.time() - start
                        if self.show_tool_events:
                            self.tool_event(tool_name, "done", is_running=False, elapsed=elapsed)
                else:
                    tool_input = json.dumps(state.get("input", {}))
                    active_tools[call_id] = (tool_name, time.time())
                    if self.show_tool_events:
                        self.tool_event(tool_name, tool_input, is_running=True)

            elif etype == "text":
                text = part.get("text", "")
                if text:
                    final_result = text

            elif etype == "step_finish":
                if part.get("reason") != "stop":
                    # Intermediate step — clear accumulated text, more turns coming
                    final_result = ""

        proc.wait()
        return final_result


def opencode_cli(
    wake_words: list[str] = ["opencode", "open code"],
    terminate_words: list[str] = ["terminate"],
    listen_duration: int | float = 5,
    continue_conversation: bool = True,
    show_tool_events: bool = True,
    model: Optional[str] = None,
    spych_kwargs: Optional[dict[str, Any]] = None,
    spych_wake_kwargs: Optional[dict[str, Any]] = None,
) -> None:
    """
    Usage:

    - Starts a wake word listener that pipes detected speech into the OpenCode CLI

    Optional:

    - `wake_words`:
        - Type: list[str]
        - What: A list of wake words that each trigger the OpenCode CLI responder
        - Default: ["opencode", "open code"]
        - Note: All wake words in this list map to the same LocalOpenCodeCLIResponder
          instance, sharing conversation history across triggers

    - `terminate_words`:
        - Type: list[str]
        - What: A list of terminate words that each trigger termination
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
        - What: Whether to print tool start/end events in the CLI as they arrive
        - Default: True

    - `model`:
        - Type: str
        - What: Model to use in provider/model format (e.g. "anthropic/claude-sonnet-4-5")
        - Default: None (uses opencode default)

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
    responder = LocalOpenCodeCLIResponder(
        spych_object=spych_object,
        continue_conversation=continue_conversation,
        listen_duration=listen_duration,
        show_tool_events=show_tool_events,
        model=model,
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