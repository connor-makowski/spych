from spych.core import Spych
from spych.orchestrator import SpychOrchestrator
from spych.responders import BaseResponder, AgentResponse
from spych.utils import resolve_cmd, StreamJsonCommand
from typing import Optional, Any
import re
import time


class LocalOpenCodeCLIResponder(BaseResponder):
    def __init__(
        self,
        spych_object: "Spych",
        continue_conversation: bool = True,
        listen_duration: int | float | str = 0,
        name: Optional[str] = None,
        show_tool_events: bool = True,
        model: Optional[str] = None,
        use_speaker: bool = True,
        speaker_voice: str = "af_heart",
        response_style: Optional[str] = None,
        **kwargs,
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
            - What: Whether to pass `--session` / `--continue` to reuse the most recent session
            - Default: True

        - `listen_duration`:
            - Type: int | float | str
            - What: How long to listen for after the wake word is detected
            - Default: 0 (Auto detect)
            - Options:
                - int | float : Record for exactly this many seconds
                - "auto" or 0 : Use Silero VAD to detect a complete utterance and
                                stop automatically when the speaker finishes

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

        - Uses `opencode run --format json` to stream newline-delimited JSON events
        - Event format (observed from real CLI output):
            - "step_start"  - model turn beginning; carries sessionID
            - "text"        - streaming text delta; part.text accumulates the response.
                              May contain inline tool XML: <function=Name>...</function></tool_call>
                              These are stripped from the final answer.
            - "step_finish" - turn done; part.reason == "stop" signals the final turn.
                              Intermediate steps (tool turns) have reason != "stop".
        - Tool calls appear as inline XML within "text" events, not as separate event types.
          They are extracted for display, then stripped from the final answer text.
        - Session continuation uses --session <id> when available, else --continue.
        - OpenCode CLI must be installed and authenticated before use.
        """
        name = name or "OpenCode"
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
        self.model = model
        self.first_call = True
        self._last_session_id: Optional[str] = None

        # Matches inline tool-call XML embedded in text deltas:
        # <function=ToolName>\n<parameter=x>val</parameter>\n</function>\n</tool_call>
        self.TOOL_CALL_RE = re.compile(
            r"<function=(\w+)>(.*?)(?:</function>|</tool_call>)",
            re.DOTALL,
        )

    def __strip_tool_calls__(self, text: str) -> str:
        """Strip inline tool-call XML from text, return clean prose only."""
        return self.TOOL_CALL_RE.sub("", text).strip()

    def respond(
        self, user_input: str, is_continuation: bool = False
    ) -> AgentResponse:
        """
        Usage:

        - Pipes the transcribed user input into `opencode run --format json`
          and returns a structured AgentResponse after all tool calls complete.
          Fires tool events live as they arrive from text deltas.

        Requires:

        - `user_input`:
            - Type: str
            - What: The transcribed text from the user's audio input

        Optional:

        - `is_continuation`:
            - Type: bool
            - What: Whether this is a continuation call after an intermediate response.
            - Default: False

        Returns:

        - `response`:
            - Type: AgentResponse
            - What: Parsed structured response (inline tool XML stripped before parse)
        """
        is_first = self.first_call
        self.first_call = False

        prompt = user_input
        if is_continuation:
            prompt = "Please continue."

        cmd = [resolve_cmd("opencode"), "run", "--format", "json"]

        if self.continue_conversation:
            if self._last_session_id:
                cmd.extend(["--session", self._last_session_id])
            elif not is_first:
                cmd.append("--continue")

        if self.model:
            cmd.extend(["--model", self.model])

        stream = StreamJsonCommand(cmd, input_text=self.format_prompt(prompt))

        # Accumulates the full streamed text for the current step.
        # Resets on each intermediate step_finish (tool turn), kept on final stop.
        accumulated_text = ""

        # Tracks tool calls seen in the current step: name -> start_time.
        # Keyed by tool name since inline XML has no unique call ID.
        active_tools: dict[str, float] = {}

        for event in stream:
            etype = event.get("type")

            # Capture session ID from any event for conversation continuity
            session_id = event.get("sessionID")
            if session_id:
                self._last_session_id = session_id

            if etype == "step_start":
                # New model turn beginning — reset accumulator for this step
                accumulated_text = ""

            elif etype == "text":
                delta = event.get("part", {}).get("text", "")
                if not delta:
                    continue

                # Check for new tool calls appearing in this delta
                if self.show_tool_events:
                    for match in self.TOOL_CALL_RE.finditer(delta):
                        tool_name = match.group(1)
                        if tool_name not in active_tools:
                            # Extract any prose before the tool call as explanation
                            preceding = delta[: match.start()]
                            explanation = self.__strip_tool_calls__(
                                preceding
                            ).strip()
                            active_tools[tool_name] = time.time()
                            self.tool_event(
                                tool_name,
                                "running",
                                is_running=True,
                                detail=explanation or None,
                            )

                accumulated_text = delta  # opencode sends full text each delta, not incremental

            elif etype == "step_finish":
                reason = event.get("part", {}).get("reason", "")

                # Close all open tool events for this step
                if self.show_tool_events:
                    for tool_name, start in list(active_tools.items()):
                        elapsed = time.time() - start
                        self.tool_event(
                            tool_name, "done", is_running=False, elapsed=elapsed
                        )
                active_tools.clear()

                if reason != "stop":
                    # Intermediate turn (tool execution step) — reset and continue
                    accumulated_text = ""

                # On reason == "stop", keep accumulated_text as the final answer

        stream.wait()

        if not accumulated_text:
            stderr_text = "".join(stream.stderr_lines).strip()
            if stderr_text:
                accumulated_text = f"Error: {stderr_text}"

        # Close any still-open tool events (e.g. if process ended unexpectedly)
        for tool_name, start in list(active_tools.items()):
            elapsed = time.time() - start
            if self.show_tool_events:
                self.tool_event(
                    tool_name, "done", is_running=False, elapsed=elapsed
                )
        active_tools.clear()

        return self.parse_output(self.__strip_tool_calls__(accumulated_text))


def opencode_cli(
    wake_words: list[str] = ["opencode", "open code"],
    terminate_words: list[str] = ["terminate"],
    listen_duration: int | float | str = 0,
    continue_conversation: bool = True,
    show_tool_events: bool = True,
    model: Optional[str] = None,
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
        - Default: 0 (Auto detect)

    - `continue_conversation`:
        - Type: bool
        - What: Whether to pass `--session` / `--continue` to reuse the most recent session
        - Default: True

    - `show_tool_events`:
        - Type: bool
        - What: Whether to print tool start/end events in the CLI as they arrive
        - Default: True

    - `model`:
        - Type: str
        - What: Model to use in provider/model format (e.g. "anthropic/claude-sonnet-4-5")
        - Default: None (uses opencode default)

    - `name`:
        - Type: str
        - What: A custom display name for the responder shown in printed messages
        - Default: None (uses "OpenCode")

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

    responder = LocalOpenCodeCLIResponder(
        spych_object=spych_object,
        continue_conversation=continue_conversation,
        listen_duration=listen_duration,
        show_tool_events=show_tool_events,
        model=model,
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
