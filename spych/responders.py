from __future__ import annotations

from spych.utils import Notify, get_response_style, get_setting, get_user, get_default_user
from spych.cli_tools import CliSpinner, NullSpinner, CliPrinter, theme, set_theme
from spych.spinners import Spinner
from spych.speaker import Speaker
from dataclasses import dataclass
from typing import Optional, Any
import json
import time
import re
from spych.dashboard import AgentDashboard

# Set theme from settings
set_theme(get_setting("theme", "dark"))


@dataclass
class AgentResponse:
    response: str
    summary: str
    requires_user_feedback: bool
    is_intermediate_response: bool = False


class BaseResponder(Notify):
    def __init__(
        self,
        spych_object: "Spych",
        listen_duration: int | float = 0,
        name: Optional[str] = None,
        spinner: Optional[CliSpinner] = None,
        response_style: str = "",
        use_speaker: bool = True,
        speaker_voice: str = "af_heart",
        speaker_backend: str = "",
        follow_up_listen_duration: int | float = 0,
        inactivity_timeout: Optional[float] = 4.0,
        dashboard: Optional[AgentDashboard] = None,
        user: Optional[str] = None,
        allow_intermediate_responses: bool = True,
        display_name: Optional[str] = None,
    ) -> None:
        """
        Usage:

        - Base class for all responders. Handles the listen-transcribe-respond cycle,
          provides a consistent interface for subclasses to implement, and includes
          a rich terminal UI and animated spinner.

        - Subclasses only need to implement `respond(user_input: str) -> str`.
          All CLI chrome (spinner, dividers, timing, response box) is handled here.

        - Public helper methods are available inside `respond()` for common
          UI needs without importing CLI internals:

            - `self.spinner.start(message)`          — restart spinner after a pause
            - `self.spinner.stop()`                  — stop spinner (e.g. before printing)
            - `self.print_info(message, color)`      — print a styled info line

        Requires:

        - `spych_object`:
            - Type: Spych
            - What: An initialized Spych instance used to record and transcribe audio
              after the wake word is detected

        Optional:

        - `listen_duration`:
            - Type: int | float
            - What: The number of seconds to listen for after the wake word is detected
            - Default: 0 (listen indefinitely until silence is detected)

        - `name`:
            - Type: str
            - What: A custom name for the responder to use in printed messages
            - Default: The class name of the responder (e.g., "Ollama")

        - `spinner`:
            - Type: CliSpinner | None
            - What: An externally-owned spinner to share across multiple responders.
              When provided (e.g. by SpychOrchestrator), all responders drive the same
              spinner so their output never interleaves. When None, a private spinner is
              created, preserving the original single-responder behaviour.
            - Default: None

        - `use_speaker`:
            - Type: bool
            - What: Whether to speak responses aloud via kokoro TTS after printing them
            - Default: True

        - `speaker_voice`:
            - Type: str
            - What: A kokoro voice ID used for all spoken responses
            - Default: "af_heart"
            - Note: American English voices use prefix `am_` or `af_`; British English
              use `bm_` or `bf_`. See spych.speaker.Speaker for the full voice list.

        - `speaker_backend`:
            - Type: str
            - What: Explicit TTS backend to use ("chatterbox" or "kokoro")
            - Default: "" (priority order: Chatterbox → Kokoro)

        - `response_style`:
            - Type: str
            - What: A style preset that shapes how the LLM formats its spoken summary.
              Named presets: concise, friendly, military, five_year_old, fast, pirate,
              news_anchor, haiku, shakespearean, robot, caveman, yoda, jarvis.
              Any other string is used verbatim as a custom instruction.
            - Default: "" (no additional style instructions)

        - `follow_up_listen_duration`:
            - Type: int | float
            - What: How long to listen for a follow-up answer after the agent speaks
              a response that contains a question. Uses the same duration semantics as
              `listen_duration`: 0 uses Silero VAD to auto-detect the end of speech;
              a positive number records for exactly that many seconds.
            - Default: 0 (VAD auto-detect)
            - Note: Only used when a follow-up is needed from the user.

        - `inactivity_timeout`:
            - Type: float | None
            - What: Seconds to wait for speech onset during a listen loop
              before pivoting back to the wake word. Only applied when using VAD-gated
              recording (duration 0).
            - Default: 4.0 (wait for 4 seconds of inactivity)

        - `dashboard`:
            - Type: AgentDashboard | None
            - What: When provided, the responder routes all lifecycle events to the
              dashboard instead of printing to stdout. The spinner is suppressed.
            - Default: None

        - `user`:
            - Type: str | None
            - What: The user name to use for tailored responses. If None, uses the
              default user from settings. If "none", no user profile is used.
            - Default: None

        - `allow_intermediate_responses`:
            - Type: bool
            - What: When True, the LLM may return `is_intermediate_response=true` to
              acknowledge a complex request before doing the actual work. The
              acknowledgment is spoken async while the continuation call runs in
              parallel. When False, the field is omitted from the schema entirely
              and a single respond() call always returns the full answer.
            - Default: True

        - `display_name`:
            - Type: str | None
            - What: A human-readable name for the responder class, used specifically
              for display in the dashboard header.
            - Default: None (falls back to class name)

        Notes:

        - Subclasses must implement `respond(user_input) -> AgentResponse`.
        - Use `self.format_prompt(prompt)` to inject the JSON schema before sending to
          the LLM, then `self.parse_output(raw)` to convert the result to AgentResponse.
        - When `use_speaker` is True, the `summary` field of AgentResponse is spoken in a
          background thread so audio playback does not block the main loop.
        - When `AgentResponse.requires_user_feedback` is True, `spoken_follow_up_loop()`
          listens for a follow-up answer and re-runs the respond cycle inline, chaining
          turns until no question remains or the user triggers the wake word.
        """
        self.spych_object = spych_object
        self.listen_duration = listen_duration
        self.name = name if name else self.__class__.__name__
        self.display_name = display_name
        self.dashboard: Optional["AgentDashboard"] = dashboard

        # Set user profile
        self.user_profile: Optional[dict] = None
        user_name = user if user else get_default_user()
        if user_name and user_name.lower() != "none":
            self.user_profile = get_user(user_name)

        # Use NullSpinner when a dashboard owns the screen; otherwise use a real
        # spinner (shared if injected by the orchestrator, private otherwise).
        _spinner: CliSpinner
        if dashboard is not None:
            _spinner = NullSpinner()
        else:
            _spinner = spinner if spinner is not None else CliSpinner()
        self.spinner: CliSpinner = _spinner
        self._start_time: float = 0.0

        self._current_user_input: str = ""
        self.use_speaker = use_speaker
        self.speaker_voice = speaker_voice
        self.speaker_backend = speaker_backend
        self.response_style = response_style
        self.style_hint = get_response_style(response_style)
        self.follow_up_listen_duration = follow_up_listen_duration
        self.inactivity_timeout = inactivity_timeout
        self.allow_intermediate_responses = allow_intermediate_responses
        self.speaker = None
        self.summary_character_limit = 300
        self._continuation_in_progress: bool = False
        self._awaiting_follow_up: bool = False
        self._current_turn_id: int = 0
        self._last_speak_turn_id: int = 0

        if use_speaker:
            self.speaker = Speaker(speaker_voice, backend=speaker_backend)

            def _on_playback_start():
                if self.dashboard is not None:
                    self.dashboard.on_status_change(
                        "speaking", turn_id=self._last_speak_turn_id
                    )

            self.speaker.on_playback_start = _on_playback_start

    # ------------------------------------------------------------------ #
    #  Public helper API — safe to call from inside respond()             #
    # ------------------------------------------------------------------ #

    @property
    def user_name(self) -> str:
        """Returns the user name from the profile or 'User'."""
        if self.user_profile:
            return self.user_profile.get("name", "User") or "User"
        return "User"

    def wait_for_next_wake_word(self, divider: bool = True) -> None:
        """
        Usage:

        - Print a divider and reset the spinner after each complete cycle to indicate
          that the responder is waiting for the next wake word. Called automatically at
          the end of `__call__` and `text_input`, but can also be called manually.

        Optional:

        - `divider`:
            - Type: bool
            - What: Whether to print a divider line before restarting the spinner
            - Default: True
        """
        if self.dashboard is not None:
            self.dashboard.on_status_change("waiting", turn_id=self._current_turn_id)
            return
        if divider:
            CliPrinter.divider()
        self.spinner.start("Waiting for wake word", spinner=Spinner.BRAILLE)

    def tool_event(
        self,
        tool_name: str,
        status: str,
        is_running: bool = False,
        elapsed: float | None = None,
        detail: str | None = None,
    ) -> None:
        """
        Usage:

        - Print a tool event message with the spinner paused and resumed automatically.
          This method stops the spinner before printing the tool event and restarts it
          afterwards to prevent interference with the output.

        Requires:

        - `tool_name`:
            - Type: str
            - What: The name of the tool being invoked

        - `status`:
            - Type: str
            - What: The status of the tool (e.g., "running", "done")

        Optional:

        - `is_running`:
            - Type: bool
            - What: Whether the tool is currently running
            - Default: False

        - `elapsed`:
            - Type: float | None
            - What: Elapsed time in seconds since the tool started (if available)
            - Default: None

        - `detail`:
            - Type: str | None
            - What: Short human-readable context shown next to the tool name,
              e.g. the file path being read, the command being run, or the
              search pattern. When updating an in-progress tool entry the
              original detail is preserved if None is passed.
            - Default: None
        """
        if self.dashboard is not None:
            self.dashboard.on_tool_event(tool_name, status, is_running=is_running, elapsed=elapsed, detail=detail)
            return
        was_running = self.spinner.stop()
        CliPrinter.tool_event(
            tool_name, status, is_running=is_running, elapsed=elapsed, detail=detail
        )
        if was_running:
            self.spinner.start()

    def print_info(self, message: str, color: str | None = None) -> None:
        """
        Usage:

        - Print a styled informational line from inside `respond()`.
          The spinner is paused automatically and resumed afterwards so
          the output line is never overwritten.

        Requires:

        - `message`:
            - Type: str
            - What: The message to display

        Optional:

        - `color`:
            - Type: str | None
            - What: ANSI escape code for the info icon. Defaults to the theme accent.
            - Default: None
        """
        if self.dashboard is not None:
            self.dashboard.on_thought(message)
            return
        was_running = self.spinner.stop()
        CliPrinter.info(message, color)
        if was_running:
            self.spinner.start()

    def print_response(self, name: str, message: str) -> None:
        """
        Usage:

        - Print a styled response message with the spinner paused and resumed automatically.
          This method stops the spinner before printing the response and restarts it
          afterwards to prevent interference with the output.

        Requires:

        - `name`:
            - Type: str
            - What: The name of the responder

        - `message`:
            - Type: str
            - What: The response message to display
        """
        if self.dashboard is not None:
            self.dashboard.on_thought(message, is_response=True)
            return
        was_running = self.spinner.stop()
        CliPrinter.print_response(name, message)
        if was_running:
            self.spinner.start()

    # ------------------------------------------------------------------ #
    #  Speaker integration                                                #
    # ------------------------------------------------------------------ #

    def format_prompt(self, prompt: str) -> str:
        """
        Usage:

        - Returns the JSON format instruction string to append to each outgoing
          prompt. Includes a style hint when `response_style` is set.

        Requires:

        - `prompt`:
            - Type: str
            - What: The raw user input to embed at the end of the formatted prompt

        Returns:

        - `formatted_prompt`:
            - Type: str
            - What: Formatted string containing JSON schema instructions, optional style
              hint, and the original user input; ready to send to the LLM
        """
        intermediate_field = (
            """- Key `is_intermediate_response` (boolean):
            - Set to true only when completing the user's request will require significant processing time (over 10 seconds, multiple tool calls, file edits, complex reasoning).
            - When true, put a brief spoken acknowledgment in `response` and `summary` (e.g. "On it — this may take a moment.").
            - If true, the calling process will present your response (or summary) to the user and then immediately tell you to continue.
            - If you find it pertinent, you can do this multiple times as you work through very long tasks, but try to limit it to one early response right away.
            - Steps to decide when to use this:
                1. Do you need to call a tool, edit files, or do any complex reasoning to complete the user's request? If no, set this to false and proceed as normal.
                2. If yes, set `is_intermediate_response` to true and provide a brief acknowledgment in `response` and `summary` to let the user know you're working on it. Don't do any actual work yet — just return this acknowledgment immediately.
                3. After you return the intermediate response, the system will present it to the user and then tell you to continue. 
                4. When you receive the continue signal, proceed with the actual work to complete the user's request, and return the final response with `is_intermediate_response` set to false.
                5. In general, you should only use this once per request, right away, but in rare cases, if the task is very long and you want to provide updates, you can use it multiple times. Just make sure to only use it for significant processing time, not for quick tool calls or reasoning steps, and try to keep the user informed with a clear acknowledgment each time.
            """
            if self.allow_intermediate_responses
            else ""
        )
        return f"""
        You must respond with a valid json object with the following keys:

        - Key: `response` (string):
            - The full text of your response, which will be printed in the terminal.
        - Key: `summary` (string):
            - A short summary of your response, that will be presented at the end of each response cycle, but may be spoken outloud.
            - Keep it casual and concise, as if you are a human summarizing your own response in a concise spoken manner.
            - Keep as short as possible while reasonably covering the core content.
            - Do not include any special characters or paths to files in this summary (in case it is read aloud).
            - Ask any follow up questions at the end of this summary. Make sure to cover the core content with this summary.
            - Do not mention any spoken instructions (like clears throat) unless to meet the instructions given by the user.
            - Keep it less than {self.summary_character_limit} characters if possible, and definitely less than {self.summary_character_limit + 100} characters.
        - Key: `requires_user_feedback` (boolean):
            - A boolean flag indicating whether your response would expect a user to follow up with additional information.
            - If true, the agent will listen for a follow-up response from the user. Make sure to ask any follow up questions last if this is true.
        {intermediate_field}

        {f"The following is information about the user you are helping. This is only for context. You should typically not reference this directly unless it is pertinent to the request stated below. START USER CONTEXT {json.dumps(self.user_profile)} END USER CONTEXT" if self.user_profile else ""}

        {"Do all tasks without considering style, however, your final response should be styled like this: " if self.style_hint else ""}
        {self.style_hint}

        Everything below is the input prompt from the user:
        =================

        {prompt}
        """

    def parse_output(self, raw_output: str) -> AgentResponse:
        """
        Usage:

        - Parses the raw text output from the LLM into an AgentResponse.
          Expects the LLM to follow the format instructions provided in `format_prompt()`.

        Requires:

        - `raw_output`:
            - Type: str
            - What: The raw text output from the LLM, expected to contain a JSON object

        Returns:

        - `response`:
            - Type: AgentResponse
            - What: Structured response extracted from the LLM's JSON output; falls back
              to echoing `raw_output` in all fields on parse failure
        """
        text = raw_output.strip()

        # Helper for reliable boolean parsing from potentially hallucinated string booleans
        def parse_bool(val: Any) -> bool:
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                return val.lower() == "true"
            return bool(val)

        # Extract JSON by finding the first '{' and using raw_decode to find the matching '}'.
        # This handles nested objects and braces inside strings correctly.
        start_idx = text.find("{")
        if start_idx != -1:
            try:
                decoder = json.JSONDecoder()
                data, _ = decoder.raw_decode(text[start_idx:])
                return AgentResponse(
                    response=str(data.get("response", raw_output)),
                    summary=str(data.get("summary", raw_output)),
                    requires_user_feedback=parse_bool(
                        data.get("requires_user_feedback", False)
                    ),
                    is_intermediate_response=parse_bool(
                        data.get("is_intermediate_response", False)
                    ),
                )
            except (json.JSONDecodeError, ValueError):
                pass

        return AgentResponse(
            response=raw_output,
            summary=raw_output,
            requires_user_feedback=False,
            is_intermediate_response=False,
        )

    # ------------------------------------------------------------------ #
    #  Extension hooks — override in subclasses for custom behaviour      #
    # ------------------------------------------------------------------ #

    def healthcheck(self) -> bool:
        """
        Usage:

        - Optional method that can be overridden to perform a health check of the responder's
          dependencies (e.g., API connectivity, model availability). If implemented, this method
          should return True if the responder is healthy and ready to respond, or False if there
          is an issue that would prevent it from functioning properly.

        """
        return True

    def respond(self, user_input: str, is_continuation: bool = False) -> AgentResponse:
        """
        Usage:

        - Called with the transcribed (or typed) user input. Must return an
          AgentResponse. All CLI chrome is handled by the base class; this method
          only needs to produce the structured response.

        - Implementations should pass the prompt through `self.format_prompt()` before
          sending to the LLM, then call `self.parse_output(raw_text)` on the result.

        - Use the public helper methods for UI feedback inside this method:

            - `self.spinner.start(message)`          — restart spinner after a pause
            - `self.spinner.stop()`                  — stop spinner (e.g. before printing)
            - `self.print_info(message, color)`      — print a styled info line

        Requires:

        - `user_input`:
            - Type: str
            - What: The transcribed or typed text from the user

        Optional:

        - `is_continuation`:
            - Type: bool
            - What: Whether this is a continuation call after an intermediate response.
              If True, the responder should typically prompt the LLM to continue its
              previous task rather than re-sending the original query.
            - Default: False

        Returns:

        - `response`:
            - Type: AgentResponse
            - What: Structured response with display text, spoken summary, and follow-up flag
        """
        raise NotImplementedError(
            "Subclasses must implement the `respond` method"
        )

    def on_before_respond(self, user_input: str) -> None:
        """
        Usage:

        - Optional lifecycle hook called immediately before `respond()`.
          Override for setup, logging, or any pre-flight work.
          The spinner is already running when this is called.

        Requires:

        - `user_input`:
            - Type: str
            - What: The enriched transcribed input (after optional clarification)
        """

    def on_after_respond(
        self, user_input: str, response: AgentResponse
    ) -> None:
        """
        Usage:

        - Optional lifecycle hook called immediately after `respond()` returns,
          before the response box is printed. Override for logging, analytics,
          caching, or post-processing.

        Requires:

        - `user_input`:
            - Type: str
            - What: The enriched transcribed input passed to `respond()`

        - `response`:
            - Type: AgentResponse
            - What: The structured response returned by `respond()`
        """

    # ------------------------------------------------------------------ #
    #  Orchestration — not intended for override; use the hooks above     #
    # ------------------------------------------------------------------ #

    def on_listen_start(self, duration: Optional[int] = None) -> None:
        """
        Usage:

        - Update the spinner message to indicate that the responder is listening for audio.

        Notes:

        - This method is called automatically at the start of each listen cycle.
        - It updates the spinner label to show the responder name and listening duration.
        """
        if self.dashboard is not None:
            self.dashboard.on_status_change("listening", turn_id=self._current_turn_id)
            return
        if duration is not None:
            listen_duration = duration
        else:
            listen_duration = self.listen_duration
        listen_string = (
            f" for {listen_duration}s" if listen_duration > 0 else ""
        )
        self.spinner.start(
            f"{theme.bold}{theme.highlight}{self.name}{theme.reset} "
            f"{theme.success}is listening{listen_string}{theme.reset}",
            spinner=Spinner.EQUALIZER,
        )

    def on_user_input(self, user_input: str, is_continuation: bool = False) -> None:
        """
        Usage:

        - Print the user input and start the spinner with verbs to indicate processing.

        Requires:

        - `user_input`:
            - Type: str
            - What: The transcribed text from the user

        Optional:

        - `is_continuation`:
            - Type: bool
            - What: Whether this is a continuation turn
            - Default: False

        Notes:

        - This method is called automatically when user input is received.
        - It prints the user input label and starts the spinner with verb animation.
        - The start time is recorded for timing purposes.
        """
        if not is_continuation:
            self._current_turn_id += 1

        self._current_user_input = user_input
        if self.dashboard is not None:
            self.dashboard.on_user_input(
                user_input, is_continuation=is_continuation, turn_id=self._current_turn_id
            )
            self.dashboard.on_status_change(
                "thinking", turn_id=self._current_turn_id
            )
            self._start_time = time.time()
            return

        if not is_continuation:
            CliPrinter.label(f"{self.user_name}:", user_input)

        self._start_time = time.time()
        if not is_continuation:
            self.spinner.start_with_verbs(
                self.name, interval=15, spinner=Spinner.ZEN
            )

    def on_response(self, response: AgentResponse) -> None:
        """
        Usage:

        - Print the final response and status information.

        Requires:

        - `response`:
            - Type: AgentResponse
            - What: The structured response returned by `respond()`

        Notes:

        - This method is called automatically after processing is complete.
        - It stops the spinner, prints the response, and shows success/failure status with elapsed time.
        """
        elapsed = time.time() - self._start_time
        if self.dashboard is not None:
            if response.response:
                self.dashboard.on_response(response, self.name, elapsed)
            
            # Only transition to waiting if we aren't about to speak,
            # no follow-up is needed, and this isn't an intermediate response.
            if not response.requires_user_feedback and not response.is_intermediate_response:
                about_to_speak = self.speaker and response.response
                if not about_to_speak:
                    self.dashboard.on_status_change("waiting", turn_id=self._current_turn_id)
            return
        self.spinner.stop()
        if response.response:
            CliPrinter.print_response(self.name, response.response)
            if (
                len(response.response) > self.summary_character_limit
                and response.summary != response.response
            ):
                CliPrinter.print_summary(response.summary)
            CliPrinter.print_status(self.name, success=True, elapsed=elapsed)
        else:
            CliPrinter.print_status(self.name, success=False, elapsed=elapsed)

    def on_terminate(self) -> None:
        """
        Usage:

        - Print a termination message and stop the spinner.

        Notes:

        - This method is called when the responder is terminated.
        - It stops the spinner and prints an informative message about the termination.
        """
        if self.dashboard is not None:
            self.dashboard.on_status_change("terminated", turn_id=self._current_turn_id)
            return
        self.spinner.stop()
        CliPrinter.info(f"{self.name} has been terminated.")

    def on_listen_end(self) -> None:
        """
        Usage:

        - Stop the spinner after listening is complete.

        Notes:

        - This method is called automatically at the end of each listen cycle.
        - It stops the spinner to indicate that listening has finished.
        """
        if self.dashboard is not None:
            return
        self.spinner.start(spinner=Spinner.BRAILLE)
        self.spinner.stop()

    def __call__(self) -> Optional[AgentResponse]:
        """
        Usage:

        - Executes one full wake-triggered cycle: listens, transcribes, responds,
          and prints.

        Returns:

        - `response`:
            - Type: AgentResponse | None
            - What: The structured response returned by `respond`, or None if
              no user input was captured or an error occurred
        """
        # Interrupt any ongoing speak event when calling the responder
        if self.speaker is not None:
            self.speaker.interrupt()

        # Set the default state for follow-up listening
        is_follow_up = False

        while True:
            duration = (
                self.follow_up_listen_duration
                if is_follow_up and self.follow_up_listen_duration != 0
                else self.listen_duration
            )
            inactivity_timeout = self.inactivity_timeout

            self.on_listen_start(duration=duration)
            user_input = self.spych_object.listen(
                duration=duration, inactivity_timeout=inactivity_timeout
            )
            self.on_listen_end()

            if not user_input:
                self.wait_for_next_wake_word(divider=False)
                return None

            current_query = user_input
            is_continuation_turn = False

            try:
                while True:
                    self.on_user_input(current_query, is_continuation=is_continuation_turn)
                    self.on_before_respond(current_query)
                    response = self.respond(current_query, is_continuation=is_continuation_turn)
                    self.on_after_respond(current_query, response)

                    # Update internal flags immediately so callbacks see the correct state
                    self._continuation_in_progress = bool(
                        self.allow_intermediate_responses
                        and response.is_intermediate_response
                    )
                    self._awaiting_follow_up = bool(response.requires_user_feedback)

                    # Unified response handling for both intermediate and final acknowledgments.
                    # We only print/speak if there is actual response text.
                    if response.response:
                        if self.speaker:
                            self.speaker.wait_for_speak()

                        self.on_response(response)

                        if self.speaker:
                            text_to_speak = (
                                response.response
                                if len(response.response) <= self.summary_character_limit
                                else response.summary
                            )
                            # Capture closure variables to fix race condition in dashboard status updates (Bug 5)
                            tid = self._current_turn_id
                            self._last_speak_turn_id = tid

                            def _on_complete():
                                if self.dashboard is not None:
                                    # Only the latest speech turn should update the status
                                    if self._last_speak_turn_id != tid:
                                        return
                                    
                                    if self._continuation_in_progress:
                                        self.dashboard.on_status_change(
                                            "thinking", turn_id=tid
                                        )
                                    elif self._awaiting_follow_up:
                                        self.dashboard.on_status_change(
                                            "listening", turn_id=tid
                                        )
                                    else:
                                        self.dashboard.on_status_change(
                                            "waiting", turn_id=tid
                                        )

                            self.speaker.speak_async(text_to_speak, on_complete=_on_complete)
                    else:
                        # For final empty responses, still call on_response to show
                        # status (e.g. failure icon) or commit the dashboard turn.
                        # This also ensures the dashboard clears the current turn input.
                        self.on_response(response)
                        if self.dashboard:
                            self.dashboard.clear_current_turn()

                    if self._continuation_in_progress:
                        is_continuation_turn = True
                        continue

                    break

            except Exception as exc:
                self._continuation_in_progress = False
                self.spinner.stop()
                if self.dashboard is not None:
                    # Notify dashboard of waiting state on error and clear turn to avoid desync (Bug 1)
                    self.dashboard.on_status_change(
                        "waiting", turn_id=self._current_turn_id
                    )
                    self.dashboard.clear_current_turn()
                else:
                    print(f"  {theme.error}✗  Error: {exc}{theme.reset}\n")
                if self.speaker:
                    self.speaker.wait_for_speak()
                self.wait_for_next_wake_word(divider=False)
                return None

            if response.requires_user_feedback:
                if self.speaker:
                    self._awaiting_follow_up = True
                    self.speaker.wait_for_speak()
                    self._awaiting_follow_up = False
                if self.dashboard is None:
                    CliPrinter.divider()
                is_follow_up = True
                # No break here; we continue the while True loop to listen for follow-up
            else:
                is_follow_up = False
                self.wait_for_next_wake_word()
                return response

    def ready_message(self, show_wait_for_wake: bool = True, **kwargs) -> None:
        """
        Formats and prints the ready message when the responder is initialized,
        showing the wake words and terminate words.

        Optional:

        - `show_wait_for_wake`:
            - Type: bool
            - What: Whether to include "Waiting for wake word" in the ready message
            - Default: True

        - `**kwargs`:
            - Type: dict
            - What: Additional keyword arguments to display in the ready message

        """
        if self.dashboard is not None:
            return
        CliPrinter.header(self.name)
        CliPrinter.kwarg_inputs(**kwargs)
        CliPrinter.empty_line()
        if show_wait_for_wake:
            self.wait_for_next_wake_word()
