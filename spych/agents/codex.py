from spych.core import Spych
from spych.orchestrator import SpychOrchestrator
from spych.responders import BaseResponder
from typing import Optional, Any
import subprocess, json, time


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

        - A responder that pipes transcribed audio into the Codex CLI (`codex exec`)
          via a subprocess and returns the final response string.

        Requires:

        - `spych_object`:
            - Type: Spych
            - What: An initialized Spych instance used to record and transcribe audio

        Optional:

        - `continue_conversation`:
            - Type: bool
            - Default: True

        - `listen_duration`:
            - Type: int | float
            - Default: 5

        - `name`:
            - Type: str
            - Default: "Codex"

        - `show_tool_events`:
            - Type: bool
            - Default: True

        Notes:

        - Uses `--json` to stream newline-delimited JSON events
        - Native tool calls are handled automatically by the CLI via `command_execution`
          items with item.started/item.completed events
        - Session continuation uses `resume --last --json` (flag order matters — placing
          `--json` before `resume --last` triggers a known Codex CLI bug where the prompt
          is parsed as SESSION_ID, causing an argument conflict error)
        - Codex CLI must be installed and authenticated before use
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

    def __run_turn__(
        self,
        user_input: str,
        is_first: bool,
        active_tools: dict[str, tuple[str, float]],
    ) -> str:
        """
        Run one `codex exec` subprocess turn and return the final response text.

        Args:
            user_input:   Prompt to send to the CLI.
            is_first:     True only on the very first ever call (skips resume).
            active_tools: Shared dict of item_id -> (tool_name, start_time).

        Returns:
            The clean final answer string from the agent.

        Notes:
            Command order for session continuation matters due to a known Codex CLI bug
            (github.com/openai/codex/issues/6717): `--json` must come AFTER
            `resume --last` / `resume <session_id>`, not before. Placing `--json`
            before `resume --last` causes Clap to treat the prompt as SESSION_ID and
            error with "the argument '--last' cannot be used with '[SESSION_ID]'".

            Correct forms:
                codex exec --json "first prompt"
                codex exec resume --last --json "follow-up"
                codex exec resume <session_id> --json "follow-up"
        """
        if self.continue_conversation and not is_first:
            if self._last_session_id:
                cmd = [
                    "codex",
                    "exec",
                    "resume",
                    self._last_session_id,
                    "--json",
                    user_input,
                ]
            else:
                cmd = [
                    "codex",
                    "exec",
                    "resume",
                    "--last",
                    "--json",
                    user_input,
                ]
        else:
            cmd = ["codex", "exec", "--json", user_input]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        final_text = ""

        for raw_line in proc.stdout:
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            etype = event.get("type")

            if etype == "thread.started":
                self._last_session_id = event.get("thread_id")

            elif etype == "item.started":
                item = event.get("item", {})
                item_id = item.get("id", "")

                if item.get("type") == "command_execution":
                    tool_name = "command_execution"
                    explanation = item.get("command", "")
                    active_tools[item_id] = (tool_name, time.time())
                    if self.show_tool_events:
                        self.tool_event(tool_name, explanation, is_running=True)

            elif etype == "item.completed":
                item = event.get("item", {})
                itype = item.get("type")
                item_id = item.get("id", "")

                if itype == "command_execution":
                    if item_id in active_tools:
                        tool_name, start = active_tools.pop(item_id)
                        elapsed = time.time() - start
                        if self.show_tool_events:
                            self.tool_event(
                                tool_name,
                                "done",
                                is_running=False,
                                elapsed=elapsed,
                            )

                elif itype == "agent_message":
                    final_text = item.get("text", "")

                elif itype == "error":
                    msg = item.get("message", "unknown error")
                    # Non-fatal model metadata warnings — skip silently
                    if "not found" in msg.lower() and "metadata" in msg.lower():
                        continue
                    final_text = f"Error: {msg}"

        proc.wait()

        # Close any still-open tool events
        for item_id, (tool_name, start) in list(active_tools.items()):
            elapsed = time.time() - start
            if self.show_tool_events:
                self.tool_event(
                    tool_name, "done", is_running=False, elapsed=elapsed
                )
        active_tools.clear()

        return final_text.strip()

    def respond(self, user_input: str) -> str:
        """
        Runs a single `codex exec` subprocess turn and returns the final answer.
        Tool calls are handled natively by the CLI — no manual re-submission needed.
        """
        is_first = self.first_call
        self.first_call = False

        active_tools: dict[str, tuple[str, float]] = {}
        return self.__run_turn__(
            user_input, is_first=is_first, active_tools=active_tools
        )


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
        - What: Whether to use `resume --last` / `resume <session_id>` to reuse the most recent session
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
        - What: Additional keyword arguments to pass to SpychWake via SpychOrchestrator
        - Default: None
    """
    spych_kwargs = {"whisper_model": "base.en", **(spych_kwargs or {})}
    spych_object = Spych(**spych_kwargs)

    responder = LocalCodexCLIResponder(
        spych_object=spych_object,
        continue_conversation=continue_conversation,
        listen_duration=listen_duration,
        show_tool_events=show_tool_events,
    )

    SpychOrchestrator(
        entries=[
            {
                "responder": responder,
                "wake_words": wake_words,
                "terminate_words": terminate_words,
            }
        ],
        spych_wake_kwargs=spych_wake_kwargs,
    ).start()