import sys
import json
import time
import importlib
import re
from spych.core import Spych
from spych.utils import resolve_cmd, StreamJsonCommand
from spych.orchestrator import SpychOrchestrator
from spych.responders import BaseResponder, AgentResponse
from typing import Optional, Any

CLAUDE_SDK_WORKER_PATH = importlib.util.find_spec(
    "spych.agents.sdk_workers.claude_sdk_worker"
).origin


def _extract_tool_detail(tool_name: str, tool_input: dict) -> str:
    """
    Usage:

    - Returns a short human-readable summary of the most relevant argument for
      a Claude Code tool call. Used to populate the ``detail`` field of tool
      events so the terminal displays e.g. ``Read  /path/to/file.py`` instead
      of a raw JSON blob.

    Requires:

    - `tool_name`:
        - Type: str
        - What: The name of the tool being invoked (e.g. "Read", "Bash").

    - `tool_input`:
        - Type: dict
        - What: The parsed input parameters for the tool call.

    Returns:

    - `detail`:
        - Type: str
        - What: A concise, display-safe string describing the key argument.
          Empty string when no meaningful detail can be extracted.
    """
    _MAP: dict[str, Any] = {
        "Read": lambda i: i.get("file_path", ""),
        "Edit": lambda i: i.get("file_path", ""),
        "Write": lambda i: i.get("file_path", ""),
        "Bash": lambda i: (i.get("command", "") or "")[:80],
        "Grep": lambda i: (
            f'"{i.get("pattern", "")}"'
            + (f' in {i.get("path", "")}' if i.get("path") else "")
        ),
        "Glob": lambda i: i.get("pattern", ""),
        "WebFetch": lambda i: i.get("url", ""),
        "WebSearch": lambda i: i.get("query", ""),
        "Agent": lambda i: i.get("description", "subagent"),
        "NotebookEdit": lambda i: i.get(
            "notebook_path", i.get("cell_type", "")
        ),
        "TodoWrite": lambda i: "updating todos",
    }
    fn = _MAP.get(tool_name)
    if fn:
        return fn(tool_input) or ""
    # Generic fallback: first non-trivial string value in the input dict
    for v in tool_input.values():
        if isinstance(v, str) and v:
            return v[:80]
    return ""


class LocalClaudeCodeSDKResponder(BaseResponder):
    def __init__(
        self,
        spych_object: "Spych",
        continue_conversation: bool = True,
        listen_duration: int | float | str = 0,
        name: str | None = None,
        setting_sources: list[str] = ["user", "project", "local"],
        show_tool_events: bool = True,
        use_speaker: bool = True,
        speaker_voice: str = "af_heart",
        response_style: Optional[str] = None,
        **kwargs,
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
            - Type: int | float | str
            - Default: 0
            - Options:
                - int | float : Record for exactly this many seconds
                - "auto" or 0 : Use Silero VAD to detect a complete utterance and
                                stop automatically when the speaker finishes

        - `name`:
            - Type: str
            - What: A custom name for the responder to use in printed messages
            - Default: "Claude"

        - `setting_sources`:
            - Type: list[str]
            - What: Which local Claude Code settings to load. Any combination of
              "user", "project", and "local".
            - Default: ["user", "project", "local"]

        - `show_tool_events`:
            - Type: bool
            - What: Whether to print tool start/end events as they arrive from the subprocess worker
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

        - Requires _sdk_worker.py in the same directory as this file
        - An ANTHROPIC_API_KEY environment variable must be set
        """
        name = name or "Claude"
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
        self.setting_sources = list(setting_sources) if setting_sources else []
        self.show_tool_events = show_tool_events
        self._first_call = True
        self._last_session_id: str | None = None

    def respond(
        self, user_input: str, is_continuation: bool = False
    ) -> AgentResponse:
        """
        Spawns _sdk_worker.py as a subprocess, writes the request payload to
        its stdin, then reads newline-delimited JSON events from its stdout.

        Tool start/end events are fired live as they arrive so the user sees
        real-time feedback. The final result is parsed into an AgentResponse.
        """
        is_first = self._first_call
        self._first_call = False

        prompt = user_input
        if is_continuation:
            prompt = "Please continue."

        payload = json.dumps(
            {
                "user_input": self.format_prompt(prompt),
                "is_first": is_first,
                "continue_conversation": self.continue_conversation,
                "last_session_id": self._last_session_id,
                "setting_sources": self.setting_sources,
            }
        )

        # print(payload)

        stream = StreamJsonCommand(
            [sys.executable, str(CLAUDE_SDK_WORKER_PATH)], input_text=payload
        )

        final_result = ""
        active_tools: dict[str, tuple[str, float]] = {}

        for event in stream:
            etype = event.get("type")

            if etype == "session":
                self._last_session_id = event.get("id")

            elif etype == "system":
                # print(raw_line)
                pass

            elif etype == "tool_start":
                tool_id = event["id"]
                tool_name = event["name"]
                raw_input = event.get("input", {})
                # print(f"Tool '{tool_name}' started with input: {json.dumps(raw_input)}")
                active_tools[tool_id] = (tool_name, time.time())
                if self.show_tool_events:
                    detail = _extract_tool_detail(tool_name, raw_input)
                    self.tool_event(
                        tool_name,
                        "running",
                        is_running=True,
                        detail=detail or None,
                    )

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

        stream.kill()

        if not final_result:
            stderr_text = "".join(stream.stderr_lines).strip()
            if stderr_text:
                final_result = f"Error: {stderr_text}"

        return self.parse_output(final_result)


def claude_code_sdk(
    wake_words: list[str] = ["claude", "clod", "cloud", "clawed"],
    terminate_words: list[str] = ["terminate"],
    listen_duration: int | float | str = 0,
    continue_conversation: bool = True,
    setting_sources: list[str] = ["user", "project", "local"],
    show_tool_events: bool = True,
    name: Optional[str] = None,
    use_speaker: bool = True,
    speaker_voice: str = "af_heart",
    response_style: Optional[str] = None,
    spych_kwargs: dict[str, any] | None = None,
    spych_wake_kwargs: dict[str, any] | None = None,
    **kwargs,
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
        - Default: 0 (Auto Detect)

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

    - `name`:
        - Type: str
        - What: A custom display name for the responder shown in printed messages
        - Default: None (uses "Claude")

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
        - What: Style preset or custom instruction for spoken output (e.g. "military", "jarvis")
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

    responder = LocalClaudeCodeSDKResponder(
        spych_object=spych_object,
        continue_conversation=continue_conversation,
        listen_duration=listen_duration,
        setting_sources=setting_sources,
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


class LocalClaudeCodeCLIResponder(BaseResponder):
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

        - A responder that pipes transcribed audio into the Claude Code CLI (`claude -p`)
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
            - What: A custom name for the responder to use in printed messages
            - Default: "Claude"

        - `show_tool_events`:
            - Type: bool
            - What: Whether to print tool start/end events as they arrive from the subprocess
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

        - Uses `--output-format stream-json --verbose`
        - The CLI may emit tool calls as raw XML inside the `result` field instead
          of executing them via the normal agent loop. When this happens, the raw
          result is re-submitted as a new `--resume` query on the same session,
          mirroring the re-query loop in claude_sdk_worker.py.
        - Intermediate assistant text (printed during tool-call turns) is printed
          live. The final clean answer is returned to the base class for printing,
          never printed directly, to avoid double-printing.
        - Claude Code CLI must be installed: npm install -g @anthropic-ai/claude-code
        """
        name = name or "Claude"
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

        # Strips inline tool call XML that the CLI embeds in assistant text blocks:
        # <function=ToolName>\n<parameter=x>val</parameter>\n</function>\n</tool_call>
        self.TOOL_CALL_RE = re.compile(
            r"<function=\w+>.*?(?:</function>|</tool_call>)",
            re.DOTALL,
        )

    def __strip_tool_calls__(self, text: str) -> str:
        """Strip inline tool-call XML from assistant text, return clean prose only."""
        return self.TOOL_CALL_RE.sub("", text).strip()

    def __run_turn__(
        self,
        user_input: str,
        is_first: bool,
        active_tools: dict[str, float],
        print_assistant: bool,
    ) -> tuple[bool, str]:
        """
        Run one `claude -p` subprocess turn.

        Args:
            user_input:      Prompt to send to the CLI.
            is_first:        True only on the very first ever call (skips --resume).
            active_tools:    Shared dict of tool_name -> start_time across turns.
            print_assistant: If True, print intermediate assistant text live.
                             Set False on the final turn to avoid double-printing
                             (the base class prints the return value of respond()).

        Returns:
            (needs_continuation, result_text)
            - needs_continuation=True means the result contained a raw
              <tool_call> block and should be re-submitted on the same session.
            - result_text is either the raw tool-call text (to re-submit) or
              the clean final answer.
        """
        cmd = [
            resolve_cmd("claude"),
            "--output-format",
            "stream-json",
            "--verbose",
        ]

        if self.continue_conversation:
            if self._last_session_id:
                cmd.extend(["--resume", self._last_session_id])
            elif not is_first:
                cmd.extend(["--resume", "latest"])

        stream = StreamJsonCommand(cmd, input_text=user_input)
        result_text = ""

        for event in stream:
            etype = event.get("type")

            if etype == "system" and event.get("subtype") == "init":
                self._last_session_id = event.get("session_id")

            elif etype == "assistant":
                content_blocks = event.get("message", {}).get("content", [])
                for block in content_blocks:
                    if block.get("type") != "text":
                        continue

                    raw_text = block.get("text", "")

                    # Fire a tool_start event for each new tool call seen
                    if self.show_tool_events:
                        for m in re.finditer(r"<function=(\w+)>", raw_text):
                            tool_name = m.group(1)
                            if tool_name not in active_tools:
                                active_tools[tool_name] = time.time()
                                preceding = raw_text[: m.start()]
                                explanation = self.__strip_tool_calls__(
                                    preceding
                                ).strip()
                                self.tool_event(
                                    tool_name,
                                    "running",
                                    is_running=True,
                                    detail=explanation or None,
                                )

                    # Only print live on intermediate (tool-call) turns.
                    # On the final turn the base class prints the return value.
                    if print_assistant:
                        clean = self.__strip_tool_calls__(raw_text)
                        if clean:
                            self.print_response(self.name, clean)

            elif etype == "result":
                result_text = event.get("result", "")

                if event.get("is_error") or event.get("subtype") == "error":
                    for tool_name, start in list(active_tools.items()):
                        if self.show_tool_events:
                            self.tool_event(
                                tool_name,
                                "done",
                                is_running=False,
                                elapsed=time.time() - start,
                            )
                    active_tools.clear()
                    stream.wait()
                    return False, f"Error: {result_text}"

        stream.wait()

        if not result_text:
            stderr_text = "".join(stream.stderr_lines).strip()
            if stderr_text:
                return False, f"Error: {stderr_text}"

        # Mirror sdk_worker: if the result contains a raw <tool_call>, re-submit
        if "</tool_call>" in result_text:
            return True, result_text

        # Clean final turn — close all open tool events
        for tool_name, start in list(active_tools.items()):
            elapsed = time.time() - start
            if self.show_tool_events:
                self.tool_event(
                    tool_name, "done", is_running=False, elapsed=elapsed
                )
        active_tools.clear()

        return False, self.__strip_tool_calls__(result_text).strip()

    def respond(
        self, user_input: str, is_continuation: bool = False
    ) -> AgentResponse:
        """
        Runs one or more `claude -p` subprocess turns until a clean result is
        received. Intermediate turns (where tool calls are in flight) print
        assistant text live. The final answer is parsed into an AgentResponse.
        """
        is_first = self.first_call
        self.first_call = False

        active_tools: dict[str, float] = {}

        prompt = user_input
        if is_continuation:
            prompt = "Please continue."

        current_input = self.format_prompt(prompt)

        while True:
            needs_continuation, result_text = self.__run_turn__(
                current_input,
                is_first=is_first,
                active_tools=active_tools,
                # Print assistant text live only on continuation turns.
                # The final turn returns its text; the base class prints it once.
                print_assistant=False,
            )
            is_first = False

            if not needs_continuation:
                return self.parse_output(result_text)

            current_input = result_text


def claude_code_cli(
    wake_words: list[str] = ["claude", "clod", "cloud", "clawed"],
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

    - Starts a wake word listener that pipes detected speech into the Claude Code CLI

    Optional:

    - `wake_words`:
        - Type: list[str]
        - Default: ["claude"]

    - `terminate_words`:
        - Type: list[str]
        - Default: ["terminate"]

    - `listen_duration`:
        - Type: int | float
        - Default: 0 (Auto)

    - `continue_conversation`:
        - Type: bool
        - Default: True

    - `show_tool_events`:
        - Type: bool
        - Default: True

    - `name`:
        - Type: str
        - What: A custom display name for the responder shown in printed messages
        - Default: None (uses "Claude")

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
        - What: Style preset or custom instruction for spoken output (e.g. "military", "jarvis")
        - Default: None

    - `spych_kwargs`:
        - Type: dict
        - Default: None

    - `spych_wake_kwargs`:
        - Type: dict
        - Default: None
    """
    spych_kwargs = {"whisper_model": "base.en", **(spych_kwargs or {})}
    spych_object = Spych(**spych_kwargs)

    responder = LocalClaudeCodeCLIResponder(
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
