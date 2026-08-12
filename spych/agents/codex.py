from spych.core import Spych
from spych.orchestrator import SpychOrchestrator
from spych.responders import BaseResponder, AgentResponse
from spych.utils import resolve_cmd, StreamJsonCommand
from typing import Optional, Any
import time


class LocalCodexCLIResponder(BaseResponder):
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
        session_id: Optional[str] = None,
        new_session: bool = False,
        **kwargs,
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
            - Type: int | float | str
            - Default: 0
            - Options:
                - int | float : Record for exactly this many seconds
                - "auto" or 0 : Use Silero VAD to detect a complete utterance and
                                stop automatically when the speaker finishes

        - `name`:
            - Type: str
            - Default: "Codex"

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

        - `session_id`:
            - Type: str | None
            - What: Spych session UUID to resume
            - Default: None

        - `new_session`:
            - Type: bool
            - What: Force a new conversation session
            - Default: False

        Notes:

        - Uses `--json` to stream newline-delimited JSON events
        - Native tool calls are handled automatically by the CLI via `command_execution`
          items with item.started/item.completed events
        - Session continuation uses `resume --last --json` (flag order matters — placing
          `--json` before `resume --last` triggers a known Codex CLI bug where the prompt
          is parsed as SESSION_ID, causing an argument conflict error)
        - Codex CLI must be installed and authenticated before use
        """
        name = name or "Codex"
        super().__init__(
            spych_object=spych_object,
            listen_duration=listen_duration,
            name=name,
            use_speaker=use_speaker,
            speaker_voice=speaker_voice,
            response_style=response_style,
            session_id=session_id,
            new_session=new_session,
            **kwargs,
        )
        self.continue_conversation = continue_conversation
        self.show_tool_events = show_tool_events
        self.first_call = True

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
        if self.continue_conversation:
            if self._last_session_id:
                cmd = [
                    resolve_cmd("codex"),
                    "exec",
                    "resume",
                    self._last_session_id,
                    "--json",
                ]
            elif not is_first:
                cmd = [
                    resolve_cmd("codex"),
                    "exec",
                    "resume",
                    "--last",
                    "--json",
                ]
            else:
                cmd = [resolve_cmd("codex"), "exec", "--json"]
        else:
            cmd = [resolve_cmd("codex"), "exec", "--json"]

        stream = StreamJsonCommand(cmd, input_text=user_input)
        final_text = ""

        try:
            for event in stream:
                etype = event.get("type")

                if etype == "thread.started":
                    thread_id = event.get("thread_id")
                    if thread_id and thread_id != self._last_session_id:
                        self._update_session_id(thread_id)

                elif etype == "item.started":
                    item = event.get("item", {})
                    item_id = item.get("id", "")

                    if item.get("type") == "command_execution":
                        tool_name = "command_execution"
                        explanation = item.get("command", "")
                        active_tools[item_id] = (tool_name, time.time())
                        if self.show_tool_events:
                            self.tool_event(
                                tool_name,
                                "running",
                                is_running=True,
                                detail=explanation or None,
                            )

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
                        if (
                            "not found" in msg.lower()
                            and "metadata" in msg.lower()
                        ):
                            continue
                        # Don't let a trailing non-fatal error clobber an
                        # answer already received via agent_message.
                        if not final_text:
                            final_text = f"Error: {msg}"
        except Exception:
            stream.kill()
            raise

        stream.wait()

        if not final_text:
            stderr_text = "".join(stream.stderr_lines).strip()
            if stderr_text:
                final_text = f"Error: {stderr_text}"

        # Close any still-open tool events
        for item_id, (tool_name, start) in list(active_tools.items()):
            elapsed = time.time() - start
            if self.show_tool_events:
                self.tool_event(
                    tool_name, "done", is_running=False, elapsed=elapsed
                )
        active_tools.clear()

        return final_text.strip()

    def respond(
        self, user_input: str, is_continuation: bool = False
    ) -> AgentResponse:
        """
        Runs a single `codex exec` subprocess turn and returns a structured
        AgentResponse. Tool calls are handled natively by the CLI.
        """
        is_first = self.first_call
        self.first_call = False

        prompt = user_input
        if is_continuation:
            prompt = "Please continue."

        active_tools: dict[str, tuple[str, float]] = {}
        raw = self.__run_turn__(
            self.format_prompt(prompt),
            is_first=is_first,
            active_tools=active_tools,
        )
        return self.parse_output(raw)


def codex_cli(
    wake_words: list[str] = ["codex"],
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
    session_id: Optional[str] = None,
    new_session: bool = False,
    **kwargs,
) -> Optional[SpychOrchestrator]:
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
        - Default: 0 (Auto detect)

    - `continue_conversation`:
        - Type: bool
        - What: Whether to use `resume --last` / `resume <session_id>` to reuse the most recent session
        - Default: True

    - `show_tool_events`:
        - Type: bool
        - What: Whether to print tool start/end events in the CLI as they arrive from the subprocess
        - Default: True

    - `name`:
        - Type: str
        - What: A custom display name for the responder shown in printed messages
        - Default: None (uses "Codex")

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

    responder = LocalCodexCLIResponder(
        spych_object=spych_object,
        continue_conversation=continue_conversation,
        listen_duration=listen_duration,
        show_tool_events=show_tool_events,
        name=name,
        use_speaker=use_speaker,
        speaker_voice=speaker_voice,
        response_style=response_style,
        session_id=session_id,
        new_session=new_session,
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
