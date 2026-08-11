from spych.core import Spych
from spych.orchestrator import SpychOrchestrator
from spych.responders import BaseResponder, AgentResponse
from spych.cli_tools import CliPrinter, theme
from spych.utils import resolve_cmd, StreamJsonCommand
from typing import Optional, Any
import time
import shutil


class LocalAntigravityCLIResponder(BaseResponder):
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

        - A responder that pipes transcribed audio into the Antigravity (agy) CLI (`agy`)
          via a subprocess and returns the final response string.
          Fires live tool-call events as the subprocess streams them.

        Requires:

        - `spych_object`:
            - Type: Spych
            - What: An initialized Spych instance used to record and transcribe audio

        Optional:

        - `continue_conversation`:
            - Type: bool
            - What: Whether to pass `--continue` or `--conversation` to reuse session
            - Default: True

        - `listen_duration`:
            - Type: int | float | str
            - What: How long to listen for after wake word is detected
            - Default: 0

        - `name`:
            - Type: str
            - What: Custom display name for responder
            - Default: "Antigravity"

        - `show_tool_events`:
            - Type: bool
            - What: Whether to print tool start/end events
            - Default: True

        - `use_speaker`:
            - Type: bool
            - What: Whether to speak responses aloud via TTS
            - Default: True

        - `speaker_voice`:
            - Type: str
            - What: Voice ID for spoken responses
            - Default: "af_heart"

        - `response_style`:
            - Type: str | None
            - What: Style preset or custom instruction for summary output
            - Default: None
        """
        name = name or "Antigravity"
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

    def _get_binary(self) -> str:
        if shutil.which("agy"):
            return "agy"
        return "antigravity"

    def healthcheck(self) -> bool:
        """
        Usage:

        - Checks that the Antigravity CLI is installed and responsive.

        Returns:

        - `is_healthy`:
            - Type: bool
            - What: True if Antigravity CLI is installed and responsive; False otherwise
        """
        binary = self._get_binary()
        if not shutil.which(binary):
            CliPrinter.info(
                "Antigravity CLI (agy) is not installed or not found on PATH.",
                color=theme.error,
            )
            CliPrinter.info(
                "Install agy CLI before running.",
                color=theme.error,
            )
            return False

        try:
            cmd = [
                resolve_cmd(binary),
                "-p",
                "return only 'true' then stop immediately.",
                "--output-format",
                "stream-json",
            ]
            stream = StreamJsonCommand(cmd)

            authenticated = False
            error_message = ""

            try:
                deadline = time.time() + 10
                while True:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        break

                    event = stream.get(timeout=remaining)
                    if event is None:
                        break

                    etype = event.get("event") or event.get("type")

                    if etype == "error":
                        error_message = event.get(
                            "message",
                            event.get("content", "unknown error"),
                        )
                        break

                    if etype in ("init", "step_update", "message", "result"):
                        authenticated = True
                        break
            finally:
                stream.kill()

            if not authenticated and not error_message:
                stderr_text = "".join(stream.stderr_lines).strip()
                if stderr_text:
                    error_message = stderr_text

            if error_message:
                CliPrinter.info(
                    f"Antigravity CLI returned an error: {error_message}",
                    color=theme.error,
                )
                return False

            if not authenticated:
                CliPrinter.info(
                    "Antigravity CLI did not respond in time.",
                    color=theme.error,
                )
                return False

            return True

        except FileNotFoundError:
            CliPrinter.info(
                "Antigravity CLI binary was found on PATH but could not be executed.",
                color=theme.error,
            )
            return False
        except Exception as e:
            CliPrinter.info(
                f"Unexpected error during Antigravity healthcheck: {e}",
                color=theme.error,
            )
            return False

    def respond(
        self, user_input: str, is_continuation: bool = False
    ) -> AgentResponse:
        """
        Usage:

        - Pipes transcribed user input into `agy -p` with `stream-json`
          and returns a structured AgentResponse after all tool calls complete.

        Requires:

        - `user_input`:
            - Type: str
            - What: Transcribed text from user's audio input

        Optional:

        - `is_continuation`:
            - Type: bool
            - What: Whether this is a continuation call
            - Default: False

        Returns:

        - `response`:
            - Type: AgentResponse
            - What: Structured response from Antigravity CLI
        """
        is_first = self.first_call
        self.first_call = False

        prompt = user_input
        if is_continuation:
            prompt = "Please continue."

        binary = self._get_binary()
        cmd = [
            resolve_cmd(binary),
            "-p",
            self.format_prompt(prompt),
            "--output-format",
            "stream-json",
        ]
        if self.continue_conversation:
            if self._last_session_id:
                cmd.extend(["--conversation", self._last_session_id])
            elif not is_first:
                cmd.extend(["--continue"])

        stream = StreamJsonCommand(cmd)

        final_result = ""
        active_tools: dict[str, tuple[str, float]] = {}

        for event in stream:
            etype = event.get("event") or event.get("type")

            if etype == "init":
                self._last_session_id = event.get("conversation_id") or (
                    event.get("init", {}).get("conversation_id")
                    if isinstance(event.get("init"), dict)
                    else None
                )

            elif etype == "step_update":
                step = event.get("step_update") or {}
                if isinstance(step, dict):
                    delta = step.get("text_delta")
                    if delta:
                        self._prev_message += delta

                    stype = step.get("step_type")
                    if stype == "tool_use":
                        tool_id = step.get("tool_id", step.get("tool_name"))
                        tool_name = step.get("tool_name", "tool")
                        params = step.get("parameters", {})
                        active_tools[tool_id] = (tool_name, time.time())
                        if self.show_tool_events:
                            detail = str(params) if params else None
                            self.tool_event(
                                tool_name,
                                "running",
                                is_running=True,
                                detail=detail,
                            )
                    elif stype == "tool_result":
                        tool_id = step.get("tool_id", step.get("tool_name"))
                        if tool_id in active_tools:
                            tool_name, start = active_tools.pop(tool_id)
                            elapsed = time.time() - start
                            if self.show_tool_events:
                                self.tool_event(
                                    tool_name,
                                    "done",
                                    is_running=False,
                                    elapsed=elapsed,
                                )

            elif etype == "result":
                res = event.get("result") or {}
                if isinstance(res, dict):
                    final_result = res.get("response") or self._prev_message
                else:
                    final_result = self._prev_message

            elif etype == "error":
                final_result = f"Error: {event.get('message', 'unknown error')}"

        stream.wait()

        if not final_result:
            final_result = self._prev_message
            if not final_result:
                stderr_text = "".join(stream.stderr_lines).strip()
                if stderr_text:
                    final_result = f"Error: {stderr_text}"

        return self.parse_output(final_result)


def antigravity_cli(
    wake_words: list[str] = ["antigravity", "gravity", "google", "gemini"],
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
    start: bool = True,
    **kwargs,
) -> Optional[SpychOrchestrator]:
    """
    Usage:

    - Starts a wake word listener that pipes detected speech into the Antigravity CLI (`agy`)

    Optional:

    - `wake_words`:
        - Type: list[str]
        - What: A list of wake words that trigger the Antigravity CLI responder
        - Default: ["antigravity", "gravity", "google", "gemini"]

    - `terminate_words`:
        - Type: list[str]
        - What: A list of terminate words that stop the listener
        - Default: ["terminate"]

    - `listen_duration`:
        - Type: int | float | str
        - What: Seconds to listen after wake word (0 = Auto detect VAD)
        - Default: 0

    - `continue_conversation`:
        - Type: bool
        - What: Whether to reuse the most recent session
        - Default: True

    - `show_tool_events`:
        - Type: bool
        - What: Whether to print tool events
        - Default: True

    - `name`:
        - Type: str
        - What: Custom display name
        - Default: None ("Antigravity")

    - `use_speaker`:
        - Type: bool
        - What: Whether to speak responses aloud via TTS
        - Default: True

    - `speaker_voice`:
        - Type: str
        - What: Voice ID for spoken responses
        - Default: "af_heart"

    - `response_style`:
        - Type: str | None
        - What: Style preset or custom instruction
        - Default: None

    - `spych_kwargs`:
        - Type: dict
        - What: Extra kwargs passed to Spych constructor
        - Default: None

    - `spych_wake_kwargs`:
        - Type: dict
        - What: Extra kwargs passed to SpychWake
        - Default: None

    - `start`:
        - Type: bool
        - What: Whether to start the orchestrator
        - Default: True
    """
    spych_kwargs = {"whisper_model": "base.en", **(spych_kwargs or {})}
    spych_object = Spych(**spych_kwargs)

    responder = LocalAntigravityCLIResponder(
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

    orchestrator = SpychOrchestrator(
        entries=[
            {
                "responder": responder,
                "wake_words": wake_words,
                "terminate_words": terminate_words,
            }
        ],
        spych_wake_kwargs=spych_wake_kwargs,
    )
    if start:
        orchestrator.start()
    return orchestrator


def agy_cli(*args, **kwargs) -> Optional[SpychOrchestrator]:
    """Alias for antigravity_cli."""
    return antigravity_cli(*args, **kwargs)
