from spych.utils import Notify, get_response_style
from spych.cli_tools import CliSpinner, CliPrinter, theme
from spych.spinners import Spinner
from spych.speaker import Speaker
from dataclasses import dataclass
from typing import Optional
import json
import time


@dataclass
class AgentResponse:
    response: str
    summary: str
    requires_user_feedback: bool


class BaseResponder(Notify):
    def __init__(
        self,
        spych_object: "Spych",
        listen_duration: int | float = 0,
        name: Optional[str] = None,
        spinner: Optional[CliSpinner] = None,
        response_style: str = "",
        use_speaker: bool = False,
        speaker_voice: str = "af_heart",
        speaker_backend: str = "",
        follow_up_listen_duration: int | float = 0,
        inactivity_timeout: Optional[float] = 8.0,
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
            - Default: False

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
            - Default: 8.0 (wait for 8 seconds of inactivity)


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
        # Accept an injected shared spinner or create a private one.
        self.spinner: CliSpinner = (
            spinner if spinner is not None else CliSpinner()
        )
        self._start_time: float = 0.0

        self._current_user_input: str = ""
        self.use_speaker = use_speaker
        self.speaker_voice = speaker_voice
        self.speaker_backend = speaker_backend
        self.response_style = response_style
        self.style_hint = get_response_style(response_style)
        self.follow_up_listen_duration = follow_up_listen_duration
        self.inactivity_timeout = inactivity_timeout
        self.speaker = None
        self.summary_character_limit = 200

        if use_speaker:
            self.speaker = Speaker(speaker_voice, backend=speaker_backend)

    # ------------------------------------------------------------------ #
    #  Public helper API — safe to call from inside respond()             #
    # ------------------------------------------------------------------ #

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
        if divider:
            CliPrinter.divider()
        self.spinner.start("Waiting for wake word", spinner=Spinner.BRAILLE)

    def tool_event(
        self,
        tool_name: str,
        status: str,
        is_running: bool = False,
        elapsed: float | None = None,
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
            - What: The status of the tool (e.g., "starting", "running", "completed", "failed")

        Optional:

        - `is_running`:
            - Type: bool
            - What: Whether the tool is currently running
            - Default: False

        - `elapsed`:
            - Type: float | None
            - What: Elapsed time in seconds since the tool started (if available)
            - Default: None
        """
        was_running = self.spinner.stop()
        CliPrinter.tool_event(
            tool_name, status, is_running=is_running, elapsed=elapsed
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

        Returns:

        - `prompt`:
            - Type: str
            - What: Formatted string ready to append to user_input
        """
        return f"""
        You must respond with a valid json object with the following keys:

        - `response`: The full text of your response, which will be printed in the terminal. 
            - Type: String
        - `summary`: A short summary of your response, that will be presented at the end of each response cycle, but may be spoken outloud. Keep as short as possible. Do not include any special characters or paths to files in this summary (in case it is read aloud).Ask any follow up questions at the end of this summary.
            - Type: String
        - `requires_user_feedback`: a boolean flag indicating whether your response contains a question or otherwise requires a user to follow up with additional information.
            - Type: Boolean
        
        {self.style_hint}

        Below is the input prompt to consider:
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
        if text.startswith("```"):
            newline = text.find("\n")
            if newline != -1:
                text = text[newline + 1 :]
            if text.endswith("```"):
                text = text[:-3].strip()
        # Try direct parse first, then fall back to extracting embedded JSON object
        for candidate in (
            text,
            text[text.find("{") : text.rfind("}") + 1] if "{" in text else "",
        ):
            try:
                data = json.loads(candidate)
                return AgentResponse(
                    response=str(data.get("response", raw_output)),
                    summary=str(data.get("summary", raw_output)),
                    requires_user_feedback=bool(
                        data.get("requires_user_feedback", False)
                    ),
                )
            except (json.JSONDecodeError, ValueError):
                continue
        return AgentResponse(
            response=raw_output,
            summary=raw_output,
            requires_user_feedback=False,
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

    def respond(self, user_input: str) -> AgentResponse:
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

    def on_user_input(self, user_input: str) -> None:
        """
        Usage:

        - Print the user input and start the spinner with verbs to indicate processing.

        Requires:

        - `user_input`:
            - Type: str
            - What: The transcribed text from the user

        Notes:

        - This method is called automatically when user input is received.
        - It prints the user input label and starts the spinner with verb animation.
        - The start time is recorded for timing purposes.
        """
        self._current_user_input = user_input
        CliPrinter.label("User:", user_input)
        self._start_time = time.time()
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

            self.on_user_input(user_input)
            try:
                self.on_before_respond(user_input)
                response = self.respond(user_input)
                self.on_after_respond(user_input, response)
            except Exception as exc:
                self.spinner.stop()
                print(f"  {theme.error}✗  Error: {exc}{theme.reset}\n")
                self.wait_for_next_wake_word(divider=False)
                return None

            self.on_response(response)

            if self.speaker and response.response:
                text_to_speak = (
                    response.response
                    if len(response.response) <= self.summary_character_limit
                    else response.summary
                )
                self.speaker.speak_async(text_to_speak)

            if response.requires_user_feedback:
                if self.speaker:
                    self.speaker.wait_for_speak()
                CliPrinter.divider()
                is_follow_up = True
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
        CliPrinter.header(self.name)
        CliPrinter.kwarg_inputs(**kwargs)
        CliPrinter.empty_line()
        if show_wait_for_wake:
            self.wait_for_next_wake_word()
