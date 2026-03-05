from spych.core import Spych
from spych.wake import SpychWake
from spych.responders import BaseResponder
from typing import Optional, Any
import subprocess, json, re, time


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
        - Codex has two tool-call modes:
            1. Native: emits `command_execution` items with item.started/item.completed
            2. Fallback: emits raw `<function=...></tool_call>` XML inside an
               `agent_message`. When this happens the raw text is re-submitted on
               the same session so the agent loop can execute the tool and continue.
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

        # Strips inline tool call XML that Codex sometimes embeds in agent_message text:
        # <function=ToolName>\n<parameter=x>val</parameter>\n</function>\n</tool_call>
        self.TOOL_CALL_RE = re.compile(
            r"<function=\w+>.*?(?:</function>|</tool_call>)",
            re.DOTALL,
        )

    def __strip_tool_calls__(self, text: str) -> str:
        """Strip inline tool-call XML from agent_message text, return clean prose only."""
        return self.TOOL_CALL_RE.sub("", text).strip()

    def __run_turn__(
        self,
        user_input: str,
        is_first: bool,
        active_tools: dict[str, tuple[str, float]],
        print_assistant: bool,
    ) -> tuple[bool, str]:
        """
        Run one `codex exec --json` subprocess turn.

        Args:
            user_input:      Prompt to send to the CLI.
            is_first:        True only on the very first ever call (skips --continue).
            active_tools:    Shared dict of item_id -> (tool_name, start_time).
            print_assistant: If True, print intermediate clean assistant text live.
                             Set False on the final turn to avoid double-printing.

        Returns:
            (needs_continuation, result_text)
            - needs_continuation=True means the agent_message contained raw
              <tool_call> XML and should be re-submitted on the same session.
            - result_text is either the raw tool-call text (to re-submit) or
              the clean final answer.
        """
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

                # Native tool: fire tool_start when the command begins executing
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
                    # Native tool completed — close the tool event
                    if item_id in active_tools:
                        tool_name, start = active_tools.pop(item_id)
                        elapsed = time.time() - start
                        if self.show_tool_events:
                            self.tool_event(tool_name, "done", is_running=False, elapsed=elapsed)

                elif itype == "agent_message":
                    raw_text = item.get("text", "")

                    # Fallback mode: inline tool-call XML embedded in the message.
                    # Fire tool events for each call found, signal re-submission.
                    if "</tool_call>" in raw_text:
                        if self.show_tool_events:
                            for m in re.finditer(r"<function=(\w+)>", raw_text):
                                tool_name = m.group(1)
                                preceding = raw_text[: m.start()]
                                explanation = self.__strip_tool_calls__(preceding).strip()
                                synthetic_id = f"fallback_{tool_name}_{item_id}"
                                active_tools[synthetic_id] = (tool_name, time.time())
                                self.tool_event(tool_name, explanation, is_running=True)
                        proc.wait()
                        return True, raw_text

                    # Normal agent message — this is the final answer
                    final_text = raw_text
                    if print_assistant:
                        clean = self.__strip_tool_calls__(raw_text)
                        if clean:
                            self.print_response(self.name, clean)

                elif itype == "error":
                    msg = item.get("message", "unknown error")
                    # Non-fatal model metadata warnings — skip silently
                    if "not found" in msg.lower() and "metadata" in msg.lower():
                        continue
                    final_text = f"Error: {msg}"

        proc.wait()

        # Clean final turn — close any still-open tool events
        for item_id, (tool_name, start) in list(active_tools.items()):
            elapsed = time.time() - start
            if self.show_tool_events:
                self.tool_event(tool_name, "done", is_running=False, elapsed=elapsed)
        active_tools.clear()

        return False, self.__strip_tool_calls__(final_text).strip()

    def respond(self, user_input: str) -> str:
        """
        Runs one or more `codex exec --json` subprocess turns until a clean
        (non-tool-call) result is received. The final answer is returned for
        the base class to print, preventing double-printing.
        """
        is_first = self.first_call
        self.first_call = False

        active_tools: dict[str, tuple[str, float]] = {}
        current_input = user_input

        while True:
            needs_continuation, result_text = self.__run_turn__(
                current_input,
                is_first=is_first,
                active_tools=active_tools,
                # Never print directly — return value is printed once by the base class
                print_assistant=False,
            )
            is_first = False

            if not needs_continuation:
                return result_text

            # Close fallback tool events before re-submitting
            for item_id in [k for k in active_tools if k.startswith("fallback_")]:
                tool_name, start = active_tools.pop(item_id)
                elapsed = time.time() - start
                if self.show_tool_events:
                    self.tool_event(tool_name, "done", is_running=False, elapsed=elapsed)

            current_input = result_text


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