from spych.core import Spych
from spych.orchestrator import SpychOrchestrator
from spych.responders import BaseResponder, AgentResponse
from spych.cli_tools import CliPrinter, theme
from spych.utils import resolve_cmd, StreamSubprocess
from typing import Optional, Any
import subprocess, json, time, shutil


class LocalGeminiCLIResponder(BaseResponder):
    def __init__(
        self,
        spych_object: "Spych",
        continue_conversation: bool = True,
        listen_duration: int | float | str = 0,
        name: Optional[str] = None,
        show_tool_events: bool = True,
        use_speaker: bool = True,
        speaker_voice: str = "af_heart",
        response_style: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Usage:

        - A responder that pipes transcribed audio into the Gemini CLI (`gemini -p`)
          via a subprocess and returns the final response string.
          Fires live tool-call events as the subprocess streams them.

        Requires:

        - `spych_object`:
            - Type: Spych
            - What: An initialized Spych instance used to record and transcribe audio

        Optional:

        - `continue_conversation`:
            - Type: bool
            - What: Whether to pass `--resume` to reuse the most recent session
            - Default: True

        - `listen_duration`:
            - Type: int | float | str
            - What: How long to listen for after the wake word is detected
            - Default: 0
            - Options:
                - int | float : Record for exactly this many seconds
                - "auto" or 0 : Use Silero VAD to detect a complete utterance and
                                stop automatically when the speaker finishes

        - `name`:
            - Type: str
            - What: A custom name for the responder to use in printed messages
            - Default: "Gemini"

        - `show_tool_events`:
            - Type: bool
            - What: Whether to print tool start/end events in the CLI as they arrive from the subprocess
            - Default: True

        - `use_speaker`:
            - Type: bool
            - What: Whether to speak responses aloud via kokoro TTS after printing them
            - Default: False

        - `speaker_voice`:
            - Type: str
            - What: A kokoro voice ID used for all spoken responses
            - Default: "af_heart"
            - Note: American English voices use prefix `am_` or `af_`; British English
              use `bm_` or `bf_`. See spych.speaker.Speaker for the full voice list.

        - `response_style`:
            - Type: str | None
            - What: Style preset or custom instruction shaping how the LLM formats its
              summary. Named presets: concise, friendly, military, five_year_old, fast,
              pirate, news_anchor, haiku, shakespearean, robot, caveman, yoda, jarvis.
              Any other string is used verbatim as a custom instruction.
            - Default: None

        Notes:

        - Uses `--output-format stream-json` to stream intermediate tool call events
        - Gemini CLI must be installed and authenticated before use
        """
        name = name or "Gemini"
        super().__init__(
            spych_object=spych_object,
            listen_duration=listen_duration,
            name=name,
            use_speaker=use_speaker,
            speaker_voice=speaker_voice,
            response_style=response_style,
            **kwargs,
        )
        self.continue_conversation = continue_conversation
        self.show_tool_events = show_tool_events
        self.first_call = True
        self._last_session_id: Optional[str] = None
        self._prev_message: str = ""
        self._prev_role: str = ""

    def healthcheck(self) -> bool:
        """
        Usage:

        - Checks that the Gemini CLI is installed and authenticated.

        Returns:

        - `is_healthy`:
            - Type: bool
            - What: True if the Gemini CLI is installed, reachable, and authenticated;
              False otherwise

        Notes:

        - Step 1: Verifies the `gemini` binary is on PATH via shutil.which
        - Step 2: Runs a minimal prompt with a short timeout to confirm
          authentication and API connectivity; parses early output for errors
        """
        # --- 1. Check the CLI binary is installed ---
        if not shutil.which("gemini"):
            CliPrinter.info(
                "Gemini CLI is not installed or not found on PATH.",
                color=theme.error,
            )
            CliPrinter.info(
                "Install it with: npm install -g @anthropic-ai/gemini-cli",
                color=theme.error,
            )
            return False

        # --- 2. Verify authentication and connectivity ---
        # Run a minimal prompt; the CLI will emit an error event early if
        # credentials are missing or the API is unreachable.
        try:
            proc = subprocess.Popen(
                [
                    resolve_cmd("gemini"),
                    "-p",
                    "return only 'true' then stop immediately.",
                    "--output-format",
                    "stream-json",
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            authenticated = False
            error_message = ""

            stream = StreamSubprocess(proc)

            try:
                # Read lines with a timeout — we only need the first few
                # events to know whether auth succeeded.
                deadline = time.time() + 10
                while True:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        break

                    raw_line = stream.get(timeout=remaining)
                    if raw_line is None:
                        break  # timed out or EOF

                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue

                    try:
                        event = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue

                    etype = event.get("type")

                    if etype == "error":
                        error_message = event.get(
                            "message",
                            event.get("content", "unknown error"),
                        )
                        break

                    # Any successful non-error event (init, message, result)
                    # means the CLI authenticated and connected to the API.
                    if etype in ("init", "message", "result"):
                        authenticated = True
                        break
            finally:
                proc.kill()
                proc.wait()

            if error_message:
                CliPrinter.info(
                    f"Gemini CLI returned an error: {error_message}",
                    color=theme.error,
                )
                if any(
                    keyword in error_message.lower()
                    for keyword in (
                        "auth",
                        "credential",
                        "login",
                        "token",
                        "api key",
                    )
                ):
                    CliPrinter.info(
                        "Run `gemini` interactively to authenticate and try again.",
                        color=theme.error,
                    )
                return False

            if not authenticated:
                CliPrinter.info(
                    "Gemini CLI did not respond in time. "
                    "Check your network connection and authentication.",
                    color=theme.error,
                )
                CliPrinter.info(
                    "Run `gemini` interactively to authenticate and try again.",
                    color=theme.error,
                )
                return False

            return True

        except FileNotFoundError:
            CliPrinter.info(
                "Gemini CLI binary was found on PATH but could not be executed.",
                color=theme.error,
            )
            return False
        except Exception as e:
            CliPrinter.info(
                f"Unexpected error during Gemini healthcheck: {e}",
                color=theme.error,
            )
            return False

    def respond(self, user_input: str) -> AgentResponse:
        """
        Usage:

        - Pipes the transcribed user input into `gemini -p` with `stream-json`
          and returns a structured AgentResponse after all tool calls complete.
          Fires tool events live as they arrive.

        Requires:

        - `user_input`:
            - Type: str
            - What: The transcribed text from the user's audio input

        Returns:

        - `response`:
            - Type: AgentResponse
            - What: Parsed structured response from Gemini CLI
        """
        is_first = self.first_call
        self.first_call = False

        cmd = [
            resolve_cmd("gemini"),
            "-p",
            self.format_prompt(user_input),
            "--output-format",
            "stream-json",
        ]

        if self.continue_conversation:
            if self._last_session_id:
                cmd.extend(["--resume", self._last_session_id])
            elif not is_first:
                cmd.extend(["--resume", "latest"])

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        final_result = ""
        # tool_id -> (name, start_time)
        active_tools: dict[str, tuple[str, float]] = {}

        for raw_line in StreamSubprocess(proc):
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            etype = event.get("type")

            if (
                len(self._prev_message) > 0
                and event.get("role", -1) != self._prev_role
                and etype != "result"
            ):
                role = (
                    self.name
                    if self._prev_role == "assistant"
                    else self._prev_role
                )
                self.print_response(role, self._prev_message)
                self._prev_message = ""
                self._prev_role = ""

            if etype == "message" and event.get("role") != "user":
                self._prev_message += event.get("content", "")
                self._prev_role = event.get("role")

            elif etype == "init":
                self._last_session_id = event.get("sessionId")

            elif etype == "tool_use":
                tool_id = event.get("tool_id", event.get("tool_name"))
                tool_name = event.get("tool_name")
                tool_input = json.dumps(event.get("parameters", {}))
                active_tools[tool_id] = (tool_name, time.time())
                if self.show_tool_events:
                    self.tool_event(tool_name, tool_input, is_running=True)

            elif etype == "tool_result":
                tool_id = event.get("tool_id", event.get("tool_name"))
                if tool_id in active_tools:
                    tool_name, start = active_tools.pop(tool_id)
                    elapsed = time.time() - start
                    if self.show_tool_events:
                        self.tool_event(
                            tool_name, "done", is_running=False, elapsed=elapsed
                        )

            elif etype == "result":
                final_result = self._prev_message
                self._prev_message = ""
                self._prev_role = ""

            elif etype == "error":
                final_result = f"Error: {event.get('message', 'unknown error')}"

        proc.wait()
        return self.parse_output(final_result)


def gemini_cli(
    wake_words: list[str] = ["gemini", "google"],
    terminate_words: list[str] = ["terminate"],
    listen_duration: int | float | str = 0,
    continue_conversation: bool = True,
    show_tool_events: bool = True,
    name: Optional[str] = None,
    use_speaker: bool = True,
    speaker_voice: str = "af_heart",
    response_style: Optional[str] = None,
    spych_kwargs: Optional[dict[str, Any]] = None,
    spych_wake_kwargs: Optional[dict[str, Any]] = None,
    **kwargs,
) -> None:
    """
    Usage:

    - Starts a wake word listener that pipes detected speech into the Gemini CLI

    Optional:

    - `wake_words`:
        - Type: list[str]
        - What: A list of wake words that each trigger the Gemini CLI responder
        - Default: ["gemini"]
        - Note: All wake words in this list map to the same LocalGeminiCLIResponder
          instance, sharing conversation history across triggers

    - `terminate_words`:
        - Type: list[str]
        - What: A list of terminate words that each trigger the termination of the Gemini CLI responder
        - Default: ["terminate"]
        - Note: All terminate words in this list map to the same LocalGeminiCLIResponder
          instance, sharing conversation history across triggers

    - `listen_duration`:
        - Type: int | float
        - What: The number of seconds to listen for after the wake word is detected
        - Default: 0 (Auto detect)

    - `continue_conversation`:
        - Type: bool
        - What: Whether to pass `--resume` to reuse the most recent session in Gemini CLI
        - Default: True

    - `show_tool_events`:
        - Type: bool
        - What: Whether to print tool start/end events in the CLI as they arrive from the subprocess
        - Default: True

    - `name`:
        - Type: str
        - What: A custom display name for the responder shown in printed messages
        - Default: None (uses "Gemini")

    - `use_speaker`:
        - Type: bool
        - What: Whether to speak responses aloud via kokoro TTS
        - Default: False

    - `speaker_voice`:
        - Type: str
        - What: Kokoro voice ID for spoken responses
        - Default: "af_heart"

    - `response_style`:
        - Type: str | None
        - What: Style preset or custom instruction for the LLM's summary output (e.g. "military", "jarvis")
        - Default: None

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

    responder = LocalGeminiCLIResponder(
        spych_object=spych_object,
        continue_conversation=continue_conversation,
        listen_duration=listen_duration,
        show_tool_events=show_tool_events,
        name=name,
        use_speaker=use_speaker,
        speaker_voice=speaker_voice,
        response_style=response_style,
        **kwargs,
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
