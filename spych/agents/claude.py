import sys, json, time, subprocess, importlib
from spych.core import Spych
from spych.wake import SpychWake
from spych.responders import BaseResponder

WORKER_PATH = importlib.util.find_spec(
    "spych.agents.sdk_workers.claude_sdk_worker"
).origin


class LocalClaudeCodeCLIResponder(BaseResponder):
    def __init__(
        self,
        spych_object: "Spych",
        continue_conversation: bool = True,
        listen_duration: int | float = 5,
        name: str | None = "Claude Code",
        setting_sources: list[str] = ["user", "project", "local"],
        show_tool_events: bool = True,
    ) -> None:
        """
        Usage:

        - A responder that pipes transcribed audio into the Claude Agent SDK
          via a subprocess worker and returns the final response string.
          Fires live tool-call events as the subprocess streams them.

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
            - Default: "Claude Code"

        - `setting_sources`:
            - Type: list[str]
            - Default: ["user", "project", "local"]

        Notes:

        - Requires _sdk_worker.py in the same directory as this file
        - An ANTHROPIC_API_KEY environment variable must be set
        """
        super().__init__(
            spych_object=spych_object,
            listen_duration=listen_duration,
            name=name,
        )
        self.continue_conversation = continue_conversation
        self.setting_sources = list(setting_sources) if setting_sources else []
        self.show_tool_events = show_tool_events
        self._first_call = True
        self._last_session_id: str | None = None

    def respond(self, user_input: str) -> str:
        """
        Spawns _sdk_worker.py as a subprocess, writes the request payload to
        its stdin, then reads newline-delimited JSON events from its stdout.

        Tool start/end events are fired live as they arrive so the user sees
        real-time feedback. The final result is returned as a string.
        """
        is_first = self._first_call
        self._first_call = False

        payload = json.dumps(
            {
                "user_input": user_input,
                "is_first": is_first,
                "continue_conversation": self.continue_conversation,
                "last_session_id": self._last_session_id,
                "setting_sources": self.setting_sources,
            }
        )

        # print(payload)

        # The worker is at spych/agents/claude_sdk_worker.py relative to the spych package root
        proc = subprocess.Popen(
            [sys.executable, str(WORKER_PATH)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        proc.stdin.write(payload + "\n")
        proc.stdin.close()

        final_result = ""
        # tool_id -> (name, start_time)
        active_tools: dict[str, tuple[str, float]] = {}

        for raw_line in proc.stdout:
            # print(raw_line, end="")  # Echo raw line for debugging
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            etype = event.get("type")

            if etype == "session":
                self._last_session_id = event.get("id")

            elif etype == "system":
                # print(raw_line)
                pass

            elif etype == "tool_start":
                tool_id = event["id"]
                tool_name = event["name"]
                tool_input = json.dumps(event.get("input", {}))
                # print(f"Tool '{tool_name}' started with input: {tool_input}")
                active_tools[tool_id] = (tool_name, time.time())
                if self.show_tool_events:
                    self.tool_event(tool_name, tool_input, is_running=True)

            elif etype == "tool_end":
                tool_id = event["id"]
                if tool_id in active_tools:
                    tool_name, start = active_tools.pop(tool_id)
                    elapsed = time.time() - start
                    if self.show_tool_events:
                        self.tool_event(
                            tool_name, "done", is_running=False, elapsed=elapsed
                        )

            elif etype == "result":
                final_result = event.get("text", "")

            elif etype == "error":
                final_result = f"Error: {event.get('text', 'unknown error')}"
        proc.wait()
        return final_result


def claude_code_cli(
    wake_words: list[str] = ["claude", "clod", "cloud", "clawed"],
    terminate_words: list[str] = ["terminate"],
    listen_duration: int | float = 5,
    continue_conversation: bool = True,
    setting_sources: list[str] = ["user", "project", "local"],
    show_tool_events: bool = True,
    spych_kwargs: dict[str, any] | None = None,
    spych_wake_kwargs: dict[str, any] | None = None,
) -> None:
    """
    Usage:

    - Starts a wake word listener that pipes detected speech into the Claude Agent SDK

    Optional:

    - `wake_words`:
        - Type: list[str]
        - What: A list of wake words that each trigger the Claude Code responder
        - Default: ["claude", "clod", "cloud", "clawed"]
        - Note: All wake words in this list map to the same LocalClaudeCodeCLIResponder
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
        - What: Whether to resume the most recent session between turns
        - Default: True

    - `setting_sources`:
        - Type: list[str]
        - What: Which local Claude Code settings to load. Any combination of
          "user", "project", and "local". When None (default), no local settings
          are loaded and the SDK runs in isolation.
        - Default: ["user", "project", "local"]
        - Example: ["user", "project", "local"] to load all local settings

    - `show_tool_events`:
        - Type: bool
        - What: Whether to print tool start/end events in the CLI as they arrive from the subprocess worker
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
    responder = LocalClaudeCodeCLIResponder(
        spych_object=spych_object,
        continue_conversation=continue_conversation,
        listen_duration=listen_duration,
        setting_sources=setting_sources,
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
    responder.ready_message(
        wake_words=wake_words, terminate_words=terminate_words
    )
    spych_wake_object.start()
